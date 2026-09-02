"""
Tests for reading a provider profile and linking it to an account.

The OAuth handshake itself is not exercised here -- that needs real client
credentials and a browser. What is tested is everything that happens after the
provider answers, which is where the account-takeover risk lives.
"""

import pytest

from app.core.oauth import (Profile, link_or_create_user, profile_from_github,
                            profile_from_google)


# =============================================================================
# GOOGLE
# =============================================================================


def test_google_verified_email_is_accepted():
    profile = profile_from_google({
        "email": "sam@example.com",
        "email_verified": True,
        "name": "Sam Rivera",
    })
    assert profile == Profile(email="sam@example.com", name="Sam Rivera")


def test_google_unverified_email_is_refused():
    """Linking happens on email, so an unverified one is a way into somebody
    else's account."""
    assert profile_from_google({
        "email": "victim@example.com",
        "email_verified": False,
        "name": "Not Them",
    }) is None


@pytest.mark.parametrize("claims", [None, {}, {"email_verified": True},
                                    {"email": "", "email_verified": True}])
def test_google_without_an_email_is_refused(claims):
    assert profile_from_google(claims) is None


def test_google_email_is_normalised():
    profile = profile_from_google({
        "email": "  Sam@Example.COM  ",
        "email_verified": True,
        "name": "Sam",
    })
    assert profile.email == "sam@example.com"


def test_google_falls_back_to_the_email_when_it_sends_no_name():
    profile = profile_from_google({
        "email": "sam@example.com", "email_verified": True,
    })
    assert profile.name == "sam"


# =============================================================================
# GITHUB
# =============================================================================


def test_github_prefers_the_primary_verified_address():
    profile = profile_from_github(
        {"login": "samr", "name": "Sam Rivera"},
        [
            {"email": "other@example.com", "primary": False, "verified": True},
            {"email": "sam@example.com", "primary": True, "verified": True},
        ],
    )
    assert profile.email == "sam@example.com"


def test_github_takes_a_verified_address_when_none_is_primary():
    profile = profile_from_github(
        {"login": "samr"},
        [{"email": "sam@example.com", "primary": False, "verified": True}],
    )
    assert profile.email == "sam@example.com"


def test_github_skips_an_unverified_primary():
    """The primary address is not automatically a trustworthy one."""
    profile = profile_from_github(
        {"login": "samr"},
        [
            {"email": "unverified@example.com", "primary": True,
             "verified": False},
            {"email": "real@example.com", "primary": False, "verified": True},
        ],
    )
    assert profile.email == "real@example.com"


def test_github_with_nothing_verified_is_refused():
    assert profile_from_github(
        {"login": "samr"},
        [{"email": "nope@example.com", "primary": True, "verified": False}],
    ) is None


@pytest.mark.parametrize("emails", [None, []])
def test_github_with_no_addresses_at_all_is_refused(emails):
    assert profile_from_github({"login": "samr"}, emails) is None


def test_github_falls_back_to_the_login_for_a_name():
    profile = profile_from_github(
        {"login": "samr", "name": None},
        [{"email": "sam@example.com", "primary": True, "verified": True}],
    )
    assert profile.name == "samr"


# =============================================================================
# LINKING
# =============================================================================


class FakeUsers:
    def __init__(self, *existing):
        self.rows = {row["email"]: row for row in existing}
        self.created = []

    def get_by_email(self, email):
        return self.rows.get(email)

    def create(self, data):
        row = dict(data, user_id=len(self.rows) + 1)
        self.rows[row["email"]] = row
        self.created.append(row)
        return row


def link(profile, users):
    return link_or_create_user(
        profile, get_user_by_email=users.get_by_email, create_user=users.create
    )


def test_a_new_email_creates_an_account_with_no_password():
    users = FakeUsers()
    account = link(Profile("sam@example.com", "Sam Rivera"), users)

    assert account["email"] == "sam@example.com"
    assert account["password_hash"] is None
    assert len(users.created) == 1


def test_a_known_email_links_to_the_account_that_already_exists():
    users = FakeUsers({
        "user_id": 7, "email": "sam@example.com", "name": "Sam",
        "password_hash": "$2b$12$realhash",
    })
    account = link(Profile("sam@example.com", "Sam From Google"), users)

    assert account["user_id"] == 7
    assert users.created == []


def test_linking_does_not_overwrite_the_existing_name_or_password():
    """Signing in with a provider is not a licence to rewrite the account."""
    users = FakeUsers({
        "user_id": 7, "email": "sam@example.com", "name": "Sam",
        "password_hash": "$2b$12$realhash",
    })
    account = link(Profile("sam@example.com", "Totally Different"), users)

    assert account["name"] == "Sam"
    assert account["password_hash"] == "$2b$12$realhash"


def test_both_providers_land_on_one_account():
    users = FakeUsers()
    first = link(Profile("sam@example.com", "Sam via Google"), users)
    second = link(Profile("sam@example.com", "samr via GitHub"), users)

    assert first["user_id"] == second["user_id"]
    assert len(users.created) == 1
