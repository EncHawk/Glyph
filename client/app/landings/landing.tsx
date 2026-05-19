import Link from 'next/link';


export default function LandingPage() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-white">
            <div className="text-center flex flex-col items-center justify-center w-[90vw] h-[100vh] bg-neutral-100
                text-2xl 
            ">
                <video autoPlay loop className='w-350 h-200 rounded-xl object-fill'>
                    <source src='/videos/dna.mp4' type='video/mp4'/>
                </video>
            </div>
        </div>
    );
}