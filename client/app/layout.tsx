import type { Metadata } from 'next';
import './globals.css';
import { Inter } from "next/font/google";
import { cn } from "@/lib/utils";

const inter = Inter({subsets:['latin'],variable:'--font-sans'});

export const metadata: Metadata = {
  title: 'Glyph AI',
  description:
    'One stop solution for all your learning, generate manim illustrations for any topic led by an agentic orchestration.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={cn("font-sans", inter.variable)}>
      <link rel="icon" type="image/x-icon" href="/favicon.ico"></link>
      <body>{children}</body>
    </html>
  );
}
