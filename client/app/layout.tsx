import type { Metadata } from 'next';
import './globals.css';
import { Inter, Geist } from "next/font/google";
import { cn } from "@/lib/utils";
import { AuthProvider } from "@/lib/auth";

const inter = Inter({subsets:['latin'],variable:'--font-sans'});
const geist = Geist({subsets:['latin'],variable:'--font-geist'});

export const metadata: Metadata = {
  title: 'Glyph AI',
  description:
    'One stop solution for all your learning, generate manim illustrations for any topic led by an agentic orchestration.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={cn("font-sans", inter.variable, geist.variable)}>
      <link rel="icon" type="image/x-icon" href="/favicon.ico"></link>
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
