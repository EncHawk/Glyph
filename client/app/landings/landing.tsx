import { Geist_Mono } from 'next/font/google';
import { EB_Garamond } from 'next/font/google';
import { Inter } from 'next/font/google';
import Prism from '../../components/Prism';
import Link from 'next/link';

const geist = Geist_Mono({ subsets: ['latin'] });
const gara = EB_Garamond({ subsets: ['latin'] });
const inter = Inter({ subsets: ['latin'] });

export default function LandingPage() {
  return (
    <main className={`${gara.className} landing-page`}>
      <div className="landing-bg">
        <Prism
          animationType="rotate"
          timeScale={0.2}
          height={3.5}
          baseWidth={5.5}
          scale={3.6}
          hueShift={0.2}
          colorFrequency={1}
          noise={0.2}
          glow={0.4}
        />
      </div>

      <div className="landing-orb landing-orb-1" />
      <div className="landing-orb landing-orb-2" />

      <section className="landing-hero">
        <h1 className="landing-title">
          <span className={`${geist.className} landing-title-accent`}>
            <i>Visualize</i>
          </span>{' '}
          your Thoughts
          <br />
          With the power of Manim
        </h1>

        <p className={`${geist.className} landing-desc`}>
          Glyph lets you bring your questions to life using custom manim illustrations,{' '}
          <br className="hidden sm:inline" />
          with something diffusion lacks : LATEX
        </p>

        <video autoPlay loop muted playsInline className="landing-video">
          <source src="/videos/dna_black.mp4" type="video/mp4" />
        </video>

        <div className="landing-actions">
          <Link href="/login" className='text-black backdrop-blur-xl p-2 bg-transparent text-shadow-md inset-shadow-sm
                shadow-sm  font-bold rounded-full cursor-pointer max-w-25 ring-[1px] ring-orange-400
                hover:scale-102 hover:inset-shadow-lg hover:ring-2 hover:ring-orange-400  hover:ring-inset-1 hover:text-orange-500 transition-al delay-75
                '>
                    Try Beta
                </Link>
                <span className='text-neutral-100'>
                    Full version is in development, try now and suggest changes.
                </span>
        </div>
      </section>
    </main>
  );
}