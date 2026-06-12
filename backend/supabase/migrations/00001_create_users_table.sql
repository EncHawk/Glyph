-- Create the users table for Glyph login
-- This is separate from the documents table used by the RAG system

CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL,
    email TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Unique constraint on (username, email) combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_email
    ON public.users (username, email);

-- Index on username for lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
    ON public.users (username);

-- Index on email for lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email
    ON public.users (email);

-- Enable RLS
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Allow anonymous inserts (login creates new users)
CREATE POLICY "Allow anonymous insert" ON public.users
    FOR INSERT
    WITH CHECK (true);

-- Allow anonymous reads (needed for login lookup)
CREATE POLICY "Allow anonymous select" ON public.users
    FOR SELECT
    USING (true);

-- Allow anonymous updates (for future profile updates)
CREATE POLICY "Allow anonymous update" ON public.users
    FOR UPDATE
    USING (true);