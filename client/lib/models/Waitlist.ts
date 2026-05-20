import mongoose from 'mongoose';

const WaitlistSchema = new mongoose.Schema({
    email: {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        trim: true,
    },
    createdAt: {
        type: Date,
        default: Date.now,
    },
});

export const Waitlist = mongoose.models.Waitlist || mongoose.model('Waitlist', WaitlistSchema);
