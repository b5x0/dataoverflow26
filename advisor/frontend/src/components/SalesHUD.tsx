import { useEffect, useState } from 'react'

interface Props {
    pitch: string | null
}

export default function SalesHUD({ pitch }: Props) {
    const [visible, setVisible] = useState(false)

    useEffect(() => {
        if (pitch) {
            setVisible(false)
            requestAnimationFrame(() => setVisible(true))
        }
    }, [pitch])

    if (!pitch) {
        return (
            <div style={{
                border: '1px dashed var(--border)',
                borderRadius: '10px',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                minHeight: '80px',
                color: 'var(--text-muted)',
            }}>
                <span style={{ fontSize: '1.2rem' }}>⚡</span>
                <span style={{ fontSize: '0.78rem', fontStyle: 'italic' }}>
                    Tactical sales pitch will appear here…
                </span>
            </div>
        )
    }

    return (
        <div
            className={`animate-hud-flash${visible ? ' animate-fade-in-up' : ''}`}
            style={{
                position: 'relative',
                background: 'linear-gradient(135deg, rgba(6,14,28,0.95), rgba(10,25,50,0.9))',
                border: '1px solid rgba(6,182,212,0.5)',
                borderRadius: '10px',
                padding: '16px 20px',
                overflow: 'hidden',
            }}
        >
            {/* Corner accent */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '3px',
                height: '100%',
                background: 'linear-gradient(180deg, var(--cyan), transparent)',
            }} />

            <div style={{
                fontSize: '0.6rem',
                fontWeight: 700,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: 'var(--cyan)',
                marginBottom: '10px',
                paddingLeft: '10px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
            }}>
                <span style={{
                    display: 'inline-block',
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: 'var(--cyan)',
                    boxShadow: '0 0 6px var(--cyan)',
                    animation: 'pulse-ring-fast 1.2s ease-in-out infinite',
                }} />
                Tactical Sales Pitch — LIVE
            </div>

            <p style={{
                fontSize: '0.88rem',
                lineHeight: 1.65,
                color: 'var(--text-primary)',
                fontWeight: 400,
                paddingLeft: '10px',
                margin: 0,
            }}>
                {pitch}
            </p>
        </div>
    )
}
