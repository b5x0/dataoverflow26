import { useEffect, useRef } from 'react'

export interface TranscriptEntry {
    id: number
    speaker: 'agent' | 'ai'
    text: string
}

interface Props {
    entries: TranscriptEntry[]
}

export default function LiveTranscription({ entries }: Props) {
    const bottomRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [entries])

    if (entries.length === 0) {
        return (
            <div style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-muted)',
                fontSize: '0.8rem',
                fontStyle: 'italic',
            }}>
                Awaiting voice session…
            </div>
        )
    }

    return (
        <div style={{
            flex: 1,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            paddingRight: '4px',
        }}>
            {entries.map(entry => (
                <div
                    key={entry.id}
                    className="animate-fade-in-up"
                    style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}
                >
                    <span
                        style={{
                            fontSize: '0.6rem',
                            fontWeight: 700,
                            letterSpacing: '0.1em',
                            textTransform: 'uppercase',
                            color: entry.speaker === 'ai' ? 'var(--purple)' : 'var(--cyan-dim)',
                        }}
                    >
                        {entry.speaker === 'ai' ? '⬡ AI Agent' : '○ Client'}
                    </span>
                    <p
                        style={{
                            fontSize: '0.83rem',
                            lineHeight: 1.55,
                            color: entry.speaker === 'ai' ? 'var(--text-primary)' : 'var(--text-secondary)',
                            borderLeft: `2px solid ${entry.speaker === 'ai' ? 'var(--purple)' : 'var(--cyan-dim)'}`,
                            paddingLeft: '10px',
                            margin: 0,
                        }}
                    >
                        {entry.text}
                    </p>
                </div>
            ))}
            <div ref={bottomRef} />
        </div>
    )
}
