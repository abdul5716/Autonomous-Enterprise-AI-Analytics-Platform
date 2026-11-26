-- Runs once, on first start of an empty Postgres volume.
-- Credentials below are for local development only.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS crm;    -- operational CRM data (customers, subscriptions, tickets)
CREATE SCHEMA IF NOT EXISTS docs;   -- document chunks, embeddings and ACL metadata
CREATE SCHEMA IF NOT EXISTS agent;  -- LangGraph checkpoints, run history, audit log

-- The agent's SQL tool connects with this role: read-only on business data, nothing else.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'analyst_ro') THEN
        CREATE ROLE analyst_ro LOGIN PASSWORD 'analyst_ro_dev';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE nimbus TO analyst_ro;
GRANT USAGE ON SCHEMA crm TO analyst_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA crm GRANT SELECT ON TABLES TO analyst_ro;
ALTER ROLE analyst_ro SET statement_timeout = '15s';
ALTER ROLE analyst_ro SET default_transaction_read_only = on;
