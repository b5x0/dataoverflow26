import { useEffect, useRef } from 'react'

interface Props {
    level: number      // 0–255 from AnalyserNode
    active: boolean    // session on?
    speaking: boolean  // AI is speaking?
}

export default function VoicePulse({ level, active, speaking }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const animRef = useRef<number>(0)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')!
        const W = canvas.width = canvas.offsetWidth * devicePixelRatio
        const H = canvas.height = canvas.offsetHeight * devicePixelRatio
        const cx = W / 2, cy = H / 2

        function draw() {
            ctx.clearRect(0, 0, W, H)

            if (!active) {
                // Idle — dim single ring
                ctx.strokeStyle = 'rgba(30,70,100,0.8)'
                ctx.lineWidth = 1.5
                ctx.beginPath()
                ctx.arc(cx, cy, W * 0.28, 0, Math.PI * 2)
                ctx.stroke()
                animRef.current = requestAnimationFrame(draw)
                return
            }

            const norm = level / 255          // 0–1
            const boost = speaking ? 1.4 : 1.0

            // Draw 3 concentric rings that expand with audio level
            const rings = [
                { r: W * 0.20 + norm * W * 0.06 * boost, opacity: 0.9, width: 2.5 },
                { r: W * 0.26 + norm * W * 0.09 * boost, opacity: 0.5, width: 1.5 },
                { r: W * 0.33 + norm * W * 0.12 * boost, opacity: 0.2, width: 1.0 },
            ]

            const color = speaking ? '168, 85, 247' : '6, 182, 212'  // purple when AI speaks

            rings.forEach(ring => {
                const grad = ctx.createRadialGradient(cx, cy, ring.r * 0.7, cx, cy, ring.r)
                grad.addColorStop(0, `rgba(${color}, ${ring.opacity})`)
                grad.addColorStop(1, `rgba(${color}, 0)`)
                ctx.strokeStyle = `rgba(${color}, ${ring.opacity})`
                ctx.lineWidth = ring.width * devicePixelRatio
                ctx.shadowBlur = 16 * norm * boost
                ctx.shadowColor = `rgba(${color}, 0.8)`
                ctx.beginPath()
                ctx.arc(cx, cy, ring.r, 0, Math.PI * 2)
                ctx.stroke()
                ctx.shadowBlur = 0
            })

            // Centre fill
            const cGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, W * 0.15)
            cGrad.addColorStop(0, `rgba(${color}, ${0.15 + norm * 0.2 * boost})`)
            cGrad.addColorStop(1, `rgba(${color}, 0)`)
            ctx.fillStyle = cGrad
            ctx.beginPath()
            ctx.arc(cx, cy, W * 0.15, 0, Math.PI * 2)
            ctx.fill()

            animRef.current = requestAnimationFrame(draw)
        }

        draw()
        return () => cancelAnimationFrame(animRef.current)
    }, [level, active, speaking])

    return (
        <canvas
            ref={canvasRef}
            style={{ width: '100%', height: '100%', display: 'block' }}
        />
    )
}
