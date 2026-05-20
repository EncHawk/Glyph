import motion from 'motion/react'
import { Geist_Mono } from 'next/font/google';
import { EB_Garamond } from 'next/font/google';
import { Inter} from 'next/font/google';
import Prism from '../../components/Prism'
import EmailInput from '../app-components/EmailInput'

const geist = Geist_Mono({ subsets: ['latin'] });
const gara = EB_Garamond({subsets:['latin']});
const inter = Inter({subsets:['latin']});

export default function LandingPage() {
    return (
    <main className={`${gara.className} min-h-screen flex flex-col items-center justify-center relative bg-neutral-950`}>
        <div className="absolute inset-0">
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
        <section className="text-center flex flex-col items-center justify-center w-full bg-transparent 
            text-neutral-100 gap-8 sm:gap-9 lg:gap-10 tracking-wider relative pt-5
        ">
            <h1
                className='text-5xl font-medium sm:font-light sm:text-6xl md:text-7xl lg:text-8xl '
            >
                <span className={` ${geist.className} tracking-tight text-transparent font-light pr-1 pl-0 bg-clip-text bg-radial-[at_50%_80%] from-red-500 to-orange-400 to-90%`}>
                    <i>Visualize</i>
                </span> your Thoughts <br/>
                With the power of Manim
            </h1>
            <p className={` text-xs sm:text-lg  text-neutral-300 font-light ${geist.className} tracking-tight text-center sm:mb-4`}>
                Glyph lets you bring your questions to life using custom manim illustrations, <br/> with something diffusion lacks : LATEX
            </p>
            <video autoPlay loop muted playsInline className='block w-75 md:w-[40vw] md:h-[40vh] rounded-xl object-fill'>
                <source src='/videos/dna_black.mp4' type='video/mp4'/>
            </video>
            <EmailInput />
        </section>
        <section>
            
        </section>
    </main>
    );
}