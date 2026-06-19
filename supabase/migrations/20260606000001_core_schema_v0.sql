-- RT-03: Core Schema v0
-- RT-02: Session and RLS Enforcement
-- Creates: profiles, conversations, messages
-- Enables RLS with per-user isolation on all tables

-- =============================================================================
-- Extensions
-- =============================================================================

create extension if not exists "pgcrypto";

-- =============================================================================
-- Tables
-- =============================================================================

create table public.profiles (
    id          uuid primary key references auth.users (id) on delete cascade,
    full_name   text,
    avatar_url  text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create table public.conversations (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references public.profiles (id) on delete cascade,
    title       text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create table public.messages (
    id              uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references public.conversations (id) on delete cascade,
    role            text not null check (role in ('user', 'assistant', 'system')),
    content         text not null,
    created_at      timestamptz not null default now()
);

-- =============================================================================
-- Indexes
-- =============================================================================

create index idx_conversations_user_id on public.conversations (user_id);
create index idx_messages_conversation_id on public.messages (conversation_id);
create index idx_messages_created_at on public.messages (conversation_id, created_at);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

alter table public.profiles enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;

-- profiles: users can only read and manage their own profile
create policy "profiles_select_own" on public.profiles
    for select to authenticated
    using (id = auth.uid());

create policy "profiles_insert_own" on public.profiles
    for insert to authenticated
    with check (id = auth.uid());

create policy "profiles_update_own" on public.profiles
    for update to authenticated
    using (id = auth.uid())
    with check (id = auth.uid());

-- conversations: users can only access their own conversations
create policy "conversations_select_own" on public.conversations
    for select to authenticated
    using (user_id = auth.uid());

create policy "conversations_insert_own" on public.conversations
    for insert to authenticated
    with check (user_id = auth.uid());

create policy "conversations_update_own" on public.conversations
    for update to authenticated
    using (user_id = auth.uid())
    with check (user_id = auth.uid());

-- messages: users can only access messages in their own conversations
create policy "messages_select_own" on public.messages
    for select to authenticated
    using (
        exists (
            select 1 from public.conversations c
            where c.id = conversation_id
              and c.user_id = auth.uid()
        )
    );

create policy "messages_insert_own" on public.messages
    for insert to authenticated
    with check (
        exists (
            select 1 from public.conversations c
            where c.id = conversation_id
              and c.user_id = auth.uid()
        )
    );

create policy "messages_update_own" on public.messages
    for update to authenticated
    using (
        exists (
            select 1 from public.conversations c
            where c.id = conversation_id
              and c.user_id = auth.uid()
        )
    )
    with check (
        exists (
            select 1 from public.conversations c
            where c.id = conversation_id
              and c.user_id = auth.uid()
        )
    );

-- =============================================================================
-- Auto-create profile on signup (trigger)
-- =============================================================================

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = ''
as $$
begin
    insert into public.profiles (id, full_name, avatar_url)
    values (
        new.id,
        coalesce(new.raw_user_meta_data ->> 'full_name', ''),
        coalesce(new.raw_user_meta_data ->> 'avatar_url', '')
    );
    return new;
end;
$$;

create trigger on_auth_user_created
    after insert on auth.users
    for each row
    execute function public.handle_new_user();
