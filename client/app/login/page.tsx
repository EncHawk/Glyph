'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';

export default function LoginPage() {
  const { login, isLoading: authLoading, user } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);

  if (user && !submitting) {
    router.replace('/fullcanvas');
    return null;
  }

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
    <div className="min-h-screen flex items-center justify-center bg-neutral-950 px-4">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm flex flex-col gap-5 bg-neutral-900 border border-neutral-800 rounded-2xl p-8"
      >
        <h1 className="text-2xl font-semibold text-neutral-100 text-center tracking-tight">
          Glyph
        </h1>
        <p className="text-sm text-neutral-400 text-center -mt-2">
          Sign in or create an account
        </p>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="username" className="text-xs text-neutral-400 font-medium">
            Username
          </label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="your_username"
            className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-100 placeholder-neutral-500 outline-none focus:border-orange-500 transition-colors"
            disabled={submitting || authLoading}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="email" className="text-xs text-neutral-400 font-medium">
            Email
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-100 placeholder-neutral-500 outline-none focus:border-orange-500 transition-colors"
            disabled={submitting || authLoading}
          />
        </div>

        <button
          type="submit"
          disabled={submitting || authLoading}
          className="bg-orange-600 hover:bg-orange-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm rounded-lg py-2.5 transition-colors cursor-pointer"
        >
          {submitting ? 'Signing in...' : 'Continue'}
        </button>

        {message && (
          <p className={`text-sm text-center ${isError ? 'text-red-400' : 'text-green-400'}`}>
            {message}
          </p>
        )}
      </form>
    </div>
  );
}