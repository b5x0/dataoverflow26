export interface ShapEntry {
    feature: string
    value: number   // positive = pushes toward prediction
    data: number   // actual feature value
}

interface Props {
    data: ShapEntry[]
    prediction: number | null
    confidence: number | null
}

const BUNDLE_NAMES: Record<number, string> = {
    0: 'Basic Essential',
    1: 'Solo Starter',
    2: 'Young Family',
    3: 'Established Family',
    4: 'Senior Shield',
    5: 'Professional Premium',
    6: 'Corporate Group',
    7: 'Premium Family Plus',
    8: 'High-Risk Managed',
    9: 'Elite Comprehensive',
}

function formatFeatureName(name: string): string {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

export default function ImpactDriverChart({ data, prediction, confidence }: Props) {
    if (prediction === null) {
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
                <div style={{ fontSize: '2rem' }}>📊</div>
                <span style={{ fontSize: '0.78rem', fontStyle: 'italic' }}>
                    Impact drivers appear after voice analysis
                </span>
            </div>
        )
    }

    const maxAbs = Math.max(...data.map(d => Math.abs(d.value)), 0.001)

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {/* Prediction Badge */}
            <div style={{
                background: 'linear-gradient(135deg, rgba(6,182,212,0.15), rgba(59,130,246,0.1))',
                border: '1px solid rgba(6,182,212,0.4)',
                borderRadius: '10px',
                padding: '12px 16px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
            }}>
                <div>
                    <div className="label">Recommended Bundle</div>
                    <div style={{
                        fontSize: '1.1rem',
                        fontWeight: 800,
                        color: 'var(--cyan-bright)',
                        marginTop: '2px',
                    }}>
                        Bundle {prediction}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                        {BUNDLE_NAMES[prediction]}
                    </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                    <div className="label">Confidence</div>
                    <div style={{
                        fontSize: '1.4rem',
                        fontWeight: 800,
                        color: (confidence ?? 0) > 60 ? 'var(--green)' : 'var(--amber)',
                        marginTop: '2px',
                        fontFamily: 'var(--font-mono)',
                    }}>
                        {confidence?.toFixed(1)}%
                    </div>
                </div>
            </div>

            {/* SHAP bars */}
            <div>
                <div className="label" style={{ marginBottom: '8px' }}>Top Impact Drivers (SHAP)</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {data.map((d, i) => {
                        const pct = (Math.abs(d.value) / maxAbs) * 100
                        const isPos = d.value >= 0
                        const color = isPos ? 'var(--cyan)' : 'var(--red)'
                        return (
                            <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    fontSize: '0.72rem',
                                }}>
                                    <span style={{ color: 'var(--text-secondary)' }}>
                                        {formatFeatureName(d.feature)}
                                    </span>
                                    <span style={{ color, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                                        {isPos ? '+' : ''}{d.value.toFixed(3)}
                                    </span>
                                </div>
                                <div style={{
                                    background: 'var(--surface)',
                                    borderRadius: '4px',
                                    height: '8px',
                                    overflow: 'hidden',
                                }}>
                                    <div style={{
                                        height: '100%',
                                        width: `${pct}%`,
                                        background: isPos
                                            ? 'linear-gradient(90deg, var(--cyan-dim), var(--cyan))'
                                            : 'linear-gradient(90deg, #991b1b, var(--red))',
                                        borderRadius: '4px',
                                        boxShadow: `0 0 8px ${color}60`,
                                        transition: 'width 0.8s cubic-bezier(0.16,1,0.3,1)',
                                    }} />
                                </div>
                                <div style={{
                                    fontSize: '0.62rem',
                                    color: 'var(--text-muted)',
                                    fontFamily: 'var(--font-mono)',
                                }}>
                                    value = {typeof d.data === 'number' ? d.data.toFixed(2) : d.data}
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>
        </div>
    )
}
