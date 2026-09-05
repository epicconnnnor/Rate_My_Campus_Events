# Oracle Cloud deployment

This deployment uses one Oracle Cloud Compute instance and Docker Compose:

- `app` runs FastAPI and applies Alembic migrations before it starts.
- `db` runs PostgreSQL 16 with pgvector, which the semantic-search migration requires.
- `proxy` exposes the site on port 80 and forwards requests to the app.

It is deliberately a single-VM deployment. Oracle's Autonomous Database is
not PostgreSQL and cannot provide pgvector, so substituting it would break the
application's schema and search feature.

## Instance

Create an Ubuntu 24.04 Compute instance (the Always Free ARM shape is suitable
when it is available). In its security list or network security group, allow
inbound TCP 80 from `0.0.0.0/0`. Do not expose port 5432.

Install Docker and the Compose plugin on the instance, then clone this
repository and enter it:

```sh
git clone https://github.com/epicconnnnor/Rate_My_Campus_Events.git
cd Rate_My_Campus_Events
```

## Secrets

Create `.env` from `.env.example`. At minimum set:

```dotenv
DATABASE_NAME=ratemycampusevents
DATABASE_USER=app
DATABASE_PASS=a-long-unique-database-password
SECRET_KEY=a-long-random-token-signing-key
RAG_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
```

Optional OAuth values can be added later. If enabled, register the deployed
site's `/auth/google/callback` and/or `/auth/github/callback` address at the
provider.

## Start and verify

```sh
docker compose -f docker-compose.oracle.yml up -d --build
curl http://localhost/healthz
```

The health check should return `{"status":"ok","database":true}`. Open
`http://<the-instance-public-ip>/` once Oracle networking allows port 80.

## HTTPS and a custom domain

After a domain points to the instance public IP, replace the `:80` line in
`Caddyfile` with the domain name (for example, `events.example.edu`) and add
`443:443` plus persistent Caddy data/config volumes to the `proxy` service.
Allow inbound TCP 443 in Oracle too. Caddy will then obtain and renew the TLS
certificate automatically.
