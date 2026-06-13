'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import PixelBlast from '../../components/PixelBlast';

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
    <div className="login-page">
      <div className="login-dither-bg">
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
      />
      </div>

      <form onSubmit={handleSubmit} className="login-card">
        <h1 className="login-heading">Glyph</h1>
        <p className="login-sub">Sign in or create an account</p>

        <div className="login-field">
          <label htmlFor="username" className="login-label">Username</label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="your_username"
            className="login-input"
            disabled={submitting || authLoading}
            autoComplete="username"
          />
        </div>

        <div className="login-field">
          <label htmlFor="email" className="login-label">Email</label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="login-input"
            disabled={submitting || authLoading}
            autoComplete="email"
          />
        </div>

        <button
          type="submit"
          disabled={submitting || authLoading}
          className="login-submit"
        >
          {submitting ? (
            <span className="login-submit-content">
              <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
              Signing in...
            </span>
          ) : 'Continue'}
        </button>

        {message && (
          <p className={`login-message ${isError ? 'login-message-error' : 'login-message-ok'}`}>
            {message}
          </p>
        )}
      </form>
    </div>
  );
}