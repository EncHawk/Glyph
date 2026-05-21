'use client';

import { useState, useRef, useEffect } from 'react';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export default function EmailInput() {
    const [email, setEmail] = useState('');
    const [width, setWidth] = useState(200);
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [message, setMessage] = useState('');
    const measureRef = useRef<HTMLSpanElement>(null);

    useEffect(() => {
        if (measureRef.current) {
            const measuredWidth = measureRef.current.offsetWidth;
            setWidth(Math.max(200, Math.min(400, measuredWidth + 32)));
        }
    }, [email]);

    const isValidEmail = (email: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

    const handleSubmit = async () => {
        if (!email.trim()) {
            setStatus('error');
            setMessage('Please enter an email');
            return;
        }

        if (!isValidEmail(email)) {
            setStatus('error');
            setMessage('Please enter a valid email');
            return;
        }

        setIsLoading(true);
        setStatus('idle');
        setMessage('');

        try {
            const res = await fetch('/api/waitlist', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email }),
            });

            const data = await res.json();

            if (res.ok) {
                setStatus('success');
                setMessage('You\'re on the waitlist!');
                setEmail('');
            } else {
                setStatus('error');
                setMessage(data.error || 'Something went wrong');
            }
        } catch {
            setStatus('error');
            setMessage('Network error, try again');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className='flex flex-col items-center gap-3'>
            <div className='flex flex-row gap-3 justify-center items-center'>
                <div className='relative'>
                    <span 
                        ref={measureRef}
                        className='invisible absolute whitespace-nowrap text-sm text-neutral-100 font-mono'
                        aria-hidden="true"
                    >
                        {email || 'garrytan@gmail.com'}
                    </span>
                    <input 
                        type="email" 
                        value={email}
                        onChange={(e) => {
                            setEmail(e.target.value);
                            if (status !== 'idle') setStatus('idle');
                        }}
                        onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                        style={{ width: `${width}px` }}
                        className='outline-none px-2 py-1 sm:px-4 sm:py-2 text-center
                        bg-transparent backdrop-blur-sm rounded-full text-sm text-neutral-100 border border-white/70
                        focus:placeholder-transparent transition-all duration-200' 
                        placeholder='garrytan@gmail.com' 
                        disabled={isLoading}
                        autoComplete='off'
                    />
                </div>
                <button 
                    onClick={handleSubmit}
                    disabled={isLoading}
                    className={`bg-orange-600 text-shadow-lg text-white text-xs sm:text-lg tracking-tight 
                    font-bold outline-none px-2 py-1 sm:px-4 sm:py-2 rounded-full ${inter.className}
                    cursor-pointer hover:bg-orange-500 hover:scale-105 transition-all delay-75
                    disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100
                `}>
                    {isLoading ? '...' : 'Join Waitlist'}
                </button>
            </div>
            {message && (
                <p className={`text-sm ${status === 'success' ? 'text-green-400' : 'text-red-400'}`}>
                    {message}
                </p>
            )}
        </div>
    );
}
