import { NextResponse } from 'next/server';
import { connectToDatabase } from '@/lib/db';
import { Waitlist } from '@/lib/models/Waitlist';

export async function POST(request: Request) {
    try {
        const { email } = await request.json();

        if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            return NextResponse.json(
                { error: 'Valid email is required' },
                { status: 400 }
            );
        }

        await connectToDatabase();

        const existing = await Waitlist.findOne({ email });
        if (existing) {
            return NextResponse.json(
                { error: 'Email already on waitlist' },
                { status: 409 }
            );
        }

        await Waitlist.create({ email });

        return NextResponse.json(
            { message: 'Successfully joined waitlist' },
            { status: 201 }
        );
    } catch (error) {
        console.error('Waitlist error:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}
