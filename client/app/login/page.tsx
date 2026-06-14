'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import PixelBlast from '../../components/PixelBlast';

export default function LoginPage() {
  const { login, user } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    if (user && !submitting) {
      router.replace('/fullcanvas');
    }
  }, [router, submitting, user]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedUser = username.trim();
    const trimmedEmail = email.trim();

    if (!trimmedUser || !trimmedEmail) {
      setMessage('Please fill in both fields');
      setIsError(true);
      return;
    }

    setSubmitting(true);
    setMessage('');
    setIsError(false);

    const result = await login(trimmedUser, trimmedEmail);

    setSubmitting(false);
    setIsError(!result.ok);
    setMessage(result.msg);

    if (result.ok) {
      router.push('/fullcanvas');
    }
  };

  return (
    <div className="isolate relative flex min-h-dvh items-center justify-center overflow-hidden bg-[#07080f] p-4">
      <div className="fixed inset-0 z-0" aria-hidden="true">
        <PixelBlast
          variant="square"
          pixelSize={4}
          color="#e7a65c"
          patternScale={2}
          patternDensity={1}
          pixelSizeJitter={0}
          enableRipples
          rippleSpeed={0.4}
          rippleThickness={0.12}
          rippleIntensityScale={1.5}
          liquid={false}
          liquidStrength={0.12}
          liquidRadius={1.2}
          liquidWobbleSpeed={5}
          speed={0.5}
          edgeFade={0.25}
          transparent
          interactive
          className="absolute inset-0"
        />
      </div>

      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-0 bg-[radial-gradient(circle_at_top,rgba(249,115,22,0.14),transparent_34%),radial-gradient(circle_at_bottom,rgba(255,255,255,0.06),transparent_30%)]"
      />

      <form
        onSubmit={handleSubmit}
        className="pointer-events-auto relative z-10 flex w-full max-w-[380px] flex-col gap-5 rounded-[1.25rem] border border-white/[0.08] bg-white/[0.03] p-10 pt-8 shadow-[0_8px_32px_rgba(0,0,0,0.4)] backdrop-blur-xl"
      >
        <h1 className="m-0 text-center text-[1.75rem] font-bold tracking-tight text-[#f0f0f5]">
          Glyph
        </h1>
        <p className="-mt-2 text-center text-[0.85rem] text-[#8b92a8]">
          Sign in or create an account
        </p>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="username" className="text-[0.72rem] font-medium uppercase tracking-wider text-[#8b92a8]">
            Username
          </label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="your_username"
            className="rounded-[0.7rem] border border-white/10 bg-white/[0.04] px-3.5 py-2.5 text-[0.9rem] text-[#e8eaf0] outline-none transition-all placeholder:text-[#4b5266] focus:border-orange-500/50 focus:bg-white/[0.06] focus:shadow-[0_0_0_3px_rgba(249,115,22,0.12)] disabled:opacity-50"
            disabled={submitting}
            autoComplete="username"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="email" className="text-[0.72rem] font-medium uppercase tracking-wider text-[#8b92a8]">
            Email
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="rounded-[0.7rem] border border-white/10 bg-white/[0.04] px-3.5 py-2.5 text-[0.9rem] text-[#e8eaf0] outline-none transition-all placeholder:text-[#4b5266] focus:border-orange-500/50 focus:bg-white/[0.06] focus:shadow-[0_0_0_3px_rgba(249,115,22,0.12)] disabled:opacity-50"
            disabled={submitting}
            autoComplete="email"
          />
        </div>

        <button
          type="submit"
          disabled={submitting}
          className="mt-1 rounded-[0.7rem] border-none bg-gradient-to-br from-[#ea580c] via-[#f97316] to-[#fb923c] px-3 py-3 text-[0.9rem] font-semibold text-white shadow-[0_4px_20px_rgba(249,115,22,0.3)] transition-all hover:-translate-y-0.5 hover:shadow-[0_6px_24px_rgba(249,115,22,0.4)] active:translate-y-0 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? (
            <span className="flex items-center justify-center gap-2">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              Signing in...
            </span>
          ) : 'Continue'}
        </button>

        {message && (
          <p className={`m-0 text-center text-[0.82rem] ${isError ? 'text-[#ff7675]' : 'text-[#19f688]'}`}>
            {message}
          </p>
        )}
      </form>
    </div>
  );
}
