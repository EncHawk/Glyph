'use client';

import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';

type User = {
  id: string;
  username: string;
  email: string;
};

type AuthContextValue = {
  user: User | null;
  isLoading: boolean;
  login: (username: string, email: string) => Promise<{ ok: boolean; msg: string }>;
  logout: () => void;
};

const AUTH_KEY = 'glyph_user';
const API_BASE = 'https://api-glyph.up.railway.app';

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const stored = localStorage.getItem(AUTH_KEY);
      if (stored) return JSON.parse(stored);
    } catch {}
    return null;
  });
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [isLoading, setIsLoading] = useState(() => {
    try {
      return !localStorage.getItem(AUTH_KEY);
    } catch {
      return true;
    }
  });

  const login = useCallback(async (username: string, email: string): Promise<{ ok: boolean; msg: string }> => {
    try {
      const res = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email }),
      });

      const data = await res.json();

      if (data.success) {
        const loggedInUser: User = {
          id: data.id,
          username: data.username,
          email: data.email,
        };
        setUser(loggedInUser);
        localStorage.setItem(AUTH_KEY, JSON.stringify(loggedInUser));
        return { ok: true, msg: data.msg };
      }

      return { ok: false, msg: data.msg || 'Login failed' };
    } catch {
      return { ok: false, msg: 'Network error' };
    }
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    localStorage.removeItem(AUTH_KEY);
  }, []);

  return (
    <AuthContext.Provider value={{ user, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}