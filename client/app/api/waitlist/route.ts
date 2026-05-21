import { NextResponse } from 'next/server';
import { connectToDatabase } from '@/lib/db';
import { Waitlist } from '@/lib/models/Waitlist';

export async function POST(request: Request) {
    try {
        const { email } = await request.json();
        const normalizedEmail = typeof email === 'string' ? email.trim().toLowerCase() : '';

        if (!normalizedEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
            return NextResponse.json(
                { error: 'Valid email is required' },
                { status: 400 }
            );
        }

        await connectToDatabase();

        const existing = await Waitlist.findOne({ email: normalizedEmail });
        if (existing) {
            return NextResponse.json(
                { error: 'Email already on waitlist' },
                { status: 409 }
            );
        }

        await Waitlist.create({ email: normalizedEmail });

        return NextResponse.json(
            { message: 'Successfully joined waitlist' },
            { status: 201 }
        );
    } catch (error) {
        console.error('Waitlist error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        return NextResponse.json(
            { error: `Failed to join waitlist: ${errorMessage}` },
            { status: 500 }
        );
    }
}
