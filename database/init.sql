-- Database initialization script for Docker
-- This script runs when the PostgreSQL container starts for the first time

-- Create the main database if it doesn't exist
SELECT 'CREATE DATABASE translation_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'translation_db')\gexec

-- Connect to the translation database
\c translation_db;

-- Run the initial schema migration
\i /docker-entrypoint-initdb.d/../migrations/001_initial_schema.sql