"""
Signing in with Google or GitHub.

Accounts are linked on email: whoever comes back from a provider with an email
we already know is that user, whether they originally signed up with a password
or through the other provider. That only works because the email is trusted, so
an address the provider has not verified is refused outright -- otherwise
anyone able to set an unverified address on a provider account could walk into
someone else's account here.

No @umass.edu restriction. Anyone can sign in.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from authlib.integrations.starlette_client import OAuth

from app.core.config import (
    GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
)

log = logging.getLogger("oauth")

GOOGLE = "google"
GITHUB = "github"
PROVIDERS = (GOOGLE, GITHUB)


@dataclass
class Profile:
    """The only two things we take from a provider."""

    email: str
    name: str


# =============================================================================
# CLIENT REGISTRY
# =============================================================================

oauth = OAuth()

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name=GOOGLE,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url=(
            "https://accounts.google.com/.well-known/openid-configuration"
        ),
        client_kwargs={"scope": "openid email profile"},
    )

if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth.register(
        name=GITHUB,
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        # user:email because a GitHub account can keep its address private, and
        # then /user returns no email at all.
        client_kwargs={"scope": "read:user user:email"},
    )


def is_configured(provider: str) -> bool:
    return bool(oauth.create_client(provider)) if provider in PROVIDERS else False


def configured_providers() -> List[str]:
    """Which buttons the login page should show."""
    return [name for name in PROVIDERS if is_configured(name)]


# =============================================================================
# READING A PROFILE
# =============================================================================


def profile_from_google(claims: Optional[Dict]) -> Optional[Profile]:
    """Google's ID token already says whether it verified the address."""
    if not claims:
        return None

    email = (claims.get("email") or "").strip().lower()
    if not email or not claims.get("email_verified"):
        log.warning("google: no verified email on the token")
        return None

    name = claims.get("name") or claims.get("given_name") or email.split("@")[0]
    return Profile(email=email, name=name)


def profile_from_github(user: Optional[Dict],
                        emails: Optional[List[Dict]]) -> Optional[Profile]:
    """GitHub keeps addresses on a separate endpoint, and only some are
    verified. Take the primary verified one, or any verified one."""
    user = user or {}
    candidates = [entry for entry in (emails or [])
                  if entry.get("verified") and entry.get("email")]

    chosen = next((entry for entry in candidates if entry.get("primary")), None)
    chosen = chosen or (candidates[0] if candidates else None)

    if not chosen:
        log.warning("github: no verified email on the account")
        return None

    email = chosen["email"].strip().lower()
    name = user.get("name") or user.get("login") or email.split("@")[0]
    return Profile(email=email, name=name)


# =============================================================================
# LINKING
# =============================================================================


def link_or_create_user(profile: Profile, *, get_user_by_email,
                        create_user) -> Dict:
    """Find the account this email already belongs to, or start one.

    An existing account is returned untouched -- its name and password are its
    own, and a provider has no business overwriting either.
    """
    existing = get_user_by_email(profile.email)
    if existing:
        return existing

    return create_user({
        "name": profile.name,
        "email": profile.email,
        # No password: this account signs in through its provider.
        "password_hash": None,
    })
