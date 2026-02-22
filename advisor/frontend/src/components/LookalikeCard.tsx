const BUNDLE_NAMES: Record<number, string> = {
    0: 'Basic Essential', 1: 'Solo Starter', 2: 'Young Family',
    3: 'Established Family', 4: 'Senior Shield', 5: 'Professional Premium',
    6: 'Corporate Group', 7: 'Premium Family Plus', 8: 'High-Risk Managed',
    9: 'Elite Comprehensive',
}

const BUNDLE_COLORS: Record<number, string> = {
    0: '#64748b', 1: '#0ea5e9', 2: '#10b981', 3: '#22c55e',
    4: '#a78bfa', 5: '#f59e0b', 6: '#3b82f6', 7: '#06b6d4',
    8: '#ef4444', 9: '#8b5cf6',
}

export interface Lookalike {
    similarity: number
    bundle: number
    income: number | null
    employment: string | null
    region: string | null
    dependents: number
    claims: number | null
}

interface Props {
    clients: Lookalike[]
}

function fmtIncome(val: number | null): string {
    if (val === null) return '—'
    return `$${(val / 1000).toFixed(0)}k`
}

export default function LookalikeCard({ clients }: Props) {
    if (clients.length === 0) {
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                flex: 1,
                gap: '8px',
                color: 'var(--text-muted)',
            }}>
                <div style={{ fontSize: '2rem' }}>👥</div>
                <span style={{ fontSize: '0.78rem', fontStyle: 'italic' }}>
                    Similar clients appear after voice analysis
                </span>
            </div>
        )
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {clients.map((c, i) => {
                const color = BUNDLE_COLORS[c.bundle] ?? '#64748b'
                return (
                    <div
                        key={i}
                        className="animate-fade-in-up"
                        style={{
                            background: 'rgba(6,14,28,0.6)',
                            border: `1px solid ${color}40`,
                            borderRadius: '10px',
                            padding: '12px',
                            position: 'relative',
                            overflow: 'hidden',
                        }}
                    >
                        {/* Similarity badge */}
                        <div style={{
                            position: 'absolute',
                            top: '10px',
                            right: '10px',
                            fontSize: '0.7rem',
                            fontWeight: 700,
                            color: '#fff',
                            background: color + '33',
                            border: `1px solid ${color}66`,
                            borderRadius: '99px',
                            padding: '2px 8px',
                            fontFamily: 'var(--font-mono)',
                        }}>
                            {c.similarity}% match
                        </div>

                        {/* Bundle label */}
                        <div style={{
                            fontSize: '0.62rem',
                            fontWeight: 700,
                            letterSpacing: '0.1em',
                            textTransform: 'uppercase',
                            color,
                            marginBottom: '4px',
                        }}>
                            Bundle {c.bundle} — {BUNDLE_NAMES[c.bundle]}
                        </div>

                        {/* Stats grid */}
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(3, 1fr)',
                            gap: '6px',
                            marginTop: '8px',
                        }}>
                            {[
                                { label: 'Income', val: fmtIncome(c.income) },
                                { label: 'Dependents', val: String(c.dependents) },
                                { label: 'Claims', val: String(c.claims ?? '—') },
                                { label: 'Employment', val: c.employment ?? '—' },
                                { label: 'Region', val: c.region ?? '—' },
                            ].map(stat => (
                                <div key={stat.label}>
                                    <div style={{
                                        fontSize: '0.58rem',
                                        color: 'var(--text-muted)',
                                        fontWeight: 600,
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.08em',
                                    }}>
                                        {stat.label}
                                    </div>
                                    <div style={{
                                        fontSize: '0.75rem',
                                        color: 'var(--text-secondary)',
                                        fontFamily: 'var(--font-mono)',
                                        marginTop: '1px',
                                    }}>
                                        {stat.val}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Match bar */}
                        <div style={{
                            marginTop: '10px',
                            background: 'var(--surface)',
                            borderRadius: '4px',
                            height: '3px',
                            overflow: 'hidden',
                        }}>
                            <div style={{
                                height: '100%',
                                width: `${c.similarity}%`,
                                background: `linear-gradient(90deg, ${color}99, ${color})`,
                                boxShadow: `0 0 6px ${color}80`,
                                transition: 'width 0.8s ease',
                            }} />
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
