import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Glyph AI',
  description: 'One stop solution for all your learning, generate manim illustrations for any topic led by an agentic orchestration.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
