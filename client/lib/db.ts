import mongoose from 'mongoose';

type MongooseCache = {
    conn: typeof mongoose | null;
    promise: Promise<typeof mongoose> | null;
};

function getMongoUri() {
    const database = process.env.MONGODB_DB?.trim() || 'waitlist';
    const authSource = process.env.MONGODB_AUTH_SOURCE?.trim() || 'admin';
    const directUri = process.env.MONGODB_URI?.trim();
    if (directUri) {
        const normalizedUri = new URL(directUri);

        if (normalizedUri.pathname === '/' || !normalizedUri.pathname) {
            normalizedUri.pathname = `/${database}`;
        }

        if (!normalizedUri.searchParams.has('appName')) {
            normalizedUri.searchParams.set('appName', database);
        }

        if (
            normalizedUri.username &&
            !normalizedUri.searchParams.has('authSource')
        ) {
            normalizedUri.searchParams.set('authSource', authSource);
        }

        return normalizedUri.toString();
    }

    const username = process.env.MONGODB_USERNAME?.trim();
    const password = process.env.MONGODB_PASSWORD?.trim();
    const cluster = process.env.MONGODB_CLUSTER?.trim();

    if (username && password && cluster) {
        const params = new URLSearchParams({
            retryWrites: 'true',
            w: 'majority',
            appName: database,
            authSource,
        });

        return `mongodb+srv://${encodeURIComponent(username)}:${encodeURIComponent(password)}@${cluster}/${database}?${params.toString()}`;
    }

    throw new Error(
        'Please define MONGODB_URI or MONGODB_USERNAME, MONGODB_PASSWORD, and MONGODB_CLUSTER.'
    );
}

let cached = (global as typeof globalThis & { mongoose?: MongooseCache }).mongoose;

if (!cached) {
    cached = (global as typeof globalThis & { mongoose?: MongooseCache }).mongoose = {
        conn: null,
        promise: null,
    };
}

export async function connectToDatabase() {
    if (cached.conn) {
        return cached.conn;
    }

    const mongoUri = getMongoUri();

    if (!cached.promise) {
        cached.promise = mongoose
            .connect(mongoUri, {
                dbName: process.env.MONGODB_DB?.trim() || undefined,
                serverSelectionTimeoutMS: 5000,
            })
            .then((mongooseInstance) => mongooseInstance)
            .catch((error) => {
                cached!.promise = null;
                throw error;
            });
    }

    cached.conn = await cached.promise;
    return cached.conn;
}
