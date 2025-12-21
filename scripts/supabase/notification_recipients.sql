-- RegionIQ: Pipeline Email Recipients (Supabase)
-- ==========================================================
-- Purpose
--   Maintain a notification routing table derived from Supabase Auth users.
--   The pipeline email stage queries *public.notification_recipients* (not auth.users)
--   to decide who receives weekly pipeline emails.
--
-- Option A (current): enabled=true by default for all new users.
-- Tightening later: change defaults / backfill rules / add allowlist logic without code refactors.
--
-- How to run
--   Paste into Supabase Dashboard → SQL Editor and execute once.

begin;

-- 1) Table: notification recipients (notification policy lives here)
create table if not exists public.notification_recipients (
  user_id uuid primary key,
  email text not null,
  enabled boolean not null default true,
  weekly_report boolean not null default true,
  failures_only boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 2) Trigger: upsert recipient record when a new auth user is created
create or replace function public.sync_notification_recipient_on_user_insert()
returns trigger
language plpgsql
security definer
set search_path = public, auth
as $$
begin
  if new.email is null then
    return new;
  end if;

  insert into public.notification_recipients
    (user_id, email, enabled, weekly_report, failures_only, created_at, updated_at)
  values
    (new.id, new.email, true, true, false, now(), now())
  on conflict (user_id) do update
    set email = excluded.email,
        updated_at = now();

  return new;
end;
$$;

drop trigger if exists trg_sync_notification_recipient_on_user_insert on auth.users;
create trigger trg_sync_notification_recipient_on_user_insert
after insert on auth.users
for each row
execute function public.sync_notification_recipient_on_user_insert();

-- 3) Trigger: keep recipient email in sync if auth.users.email changes
create or replace function public.sync_notification_recipient_on_user_update_email()
returns trigger
language plpgsql
security definer
set search_path = public, auth
as $$
begin
  if new.email is null then
    return new;
  end if;

  update public.notification_recipients
    set email = new.email,
        updated_at = now()
  where user_id = new.id;

  return new;
end;
$$;

drop trigger if exists trg_sync_notification_recipient_on_user_update_email on auth.users;
create trigger trg_sync_notification_recipient_on_user_update_email
after update of email on auth.users
for each row
execute function public.sync_notification_recipient_on_user_update_email();

-- 4) Backfill existing users into recipient table
insert into public.notification_recipients
  (user_id, email, enabled, weekly_report, failures_only, created_at, updated_at)
select
  u.id,
  u.email,
  true,
  true,
  false,
  now(),
  now()
from auth.users u
where u.email is not null
on conflict (user_id) do update
  set email = excluded.email,
      updated_at = now();

commit;

-- 5) Sanity checks
-- select count(*) as auth_user_count from auth.users;
-- select count(*) as recipient_count from public.notification_recipients;
-- select * from public.notification_recipients order by created_at desc limit 10;


