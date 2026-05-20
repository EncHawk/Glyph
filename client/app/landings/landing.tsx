import Link from 'next/link';
import { Geist_Mono } from 'next/font/google';
import { EB_Garamond } from 'next/font/google';
import { Inter} from 'next/font/google';
import Prism from '../../components/Prism'

const geist = Geist_Mono({ subsets: ['latin'] });
const gara = EB_Garamond({subsets:['latin']});
const edu = Inter({subsets:['latin']});

export default function LandingPage() {
    return (
    <main className={`${gara.className} min-h-screen flex items-center justify-center relative bg-neutral-950`}>
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
            text-neutral-100 gap-12 sm:gap-9 lg:gap-10 tracking-wider relative pt-6 sm:pt-10 lg:pt-30
        ">
            <h1
                className='text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-light'
            >
                <span className={` ${geist.className} tracking-tight text-transparent font-light pr-1 pl-0 bg-clip-text bg-radial-[at_50%_80%] from-red-500 to-orange-500 to-90%`}>
                    <i>Visualize</i>
                </span> your Thoughts <br/>
                With the power of Manim
            </h1>
            <video autoPlay loop muted playsInline className='block w-60 md:w-[30vw] md:h-[30vh] rounded-xl object-fill'>
                <source src='/videos/knn_black.mp4' type='video/mp4'/>
            </video>
        </section>
    </main>
    );
}