interface FormData {
    Estimated_Annual_Income: string
    Employment_Status: string
    Region_Code: string
    Adult_Dependents: string
    Child_Dependents: string
    Infant_Dependents: string
    Previous_Claims_Filed: string
    Years_Without_Claims: string
    Deductible_Tier: string
    Acquisition_Channel: string
    Payment_Schedule: string
    Policy_Start_Month: string
    Existing_Policyholder: string
    Previous_Policy_Duration_Months: string
    Vehicles_on_Policy: string
    Custom_Riders_Requested: string
    Grace_Period_Extensions: string
    Policy_Cancelled_Post_Purchase: string
}

interface Props {
    data: FormData
    autoFilled: Set<string>
    onChange: (field: string, value: string) => void
    onSubmit: () => void
    loading: boolean
}

const FIELDS: Array<{
    key: keyof FormData
    label: string
    type: 'number' | 'text' | 'select'
    options?: string[]
}> = [
        { key: 'Estimated_Annual_Income', label: 'Annual Income ($)', type: 'number' },
        {
            key: 'Employment_Status', label: 'Employment Status', type: 'select',
            options: ['Full-Time', 'Part-Time', 'Self-Employed', 'Retired', 'Unemployed']
        },
        { key: 'Region_Code', label: 'Region Code', type: 'text' },
        { key: 'Adult_Dependents', label: 'Adult Dependents', type: 'number' },
        { key: 'Child_Dependents', label: 'Child Dependents', type: 'number' },
        { key: 'Infant_Dependents', label: 'Infant Dependents', type: 'number' },
        { key: 'Previous_Claims_Filed', label: 'Previous Claims Filed', type: 'number' },
        { key: 'Years_Without_Claims', label: 'Years Without Claims', type: 'number' },
        {
            key: 'Deductible_Tier', label: 'Deductible Tier', type: 'select',
            options: ['Low', 'Medium', 'High']
        },
        {
            key: 'Acquisition_Channel', label: 'Acquisition Channel', type: 'select',
            options: ['Online', 'Agent', 'Broker', 'Direct', 'Referral']
        },
        {
            key: 'Payment_Schedule', label: 'Payment Schedule', type: 'select',
            options: ['Monthly', 'Quarterly', 'Annual']
        },
        {
            key: 'Policy_Start_Month', label: 'Policy Start Month', type: 'select',
            options: ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
        },
        {
            key: 'Existing_Policyholder', label: 'Existing Policyholder', type: 'select',
            options: ['0', '1']
        },
        { key: 'Previous_Policy_Duration_Months', label: 'Prior Policy Duration (mo)', type: 'number' },
        { key: 'Vehicles_on_Policy', label: 'Vehicles on Policy', type: 'number' },
        { key: 'Custom_Riders_Requested', label: 'Custom Riders', type: 'number' },
        { key: 'Grace_Period_Extensions', label: 'Grace Extensions', type: 'number' },
        {
            key: 'Policy_Cancelled_Post_Purchase', label: 'Policy Cancelled?', type: 'select',
            options: ['0', '1']
        },
    ]

export type { FormData }

export default function ManualOverrideForm({ data, autoFilled, onChange, onSubmit, loading }: Props) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', height: '100%' }}>
            {/* Auto-fill banner */}
            {autoFilled.size > 0 && (
                <div style={{
                    background: 'rgba(6,182,212,0.08)',
                    border: '1px solid rgba(6,182,212,0.3)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '0.72rem',
                    color: 'var(--cyan)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                }}>
                    <span>✦</span>
                    {autoFilled.size} field{autoFilled.size > 1 ? 's' : ''} auto-filled from voice · Review before submitting
                </div>
            )}

            {/* Field grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '10px',
                flex: 1,
                overflowY: 'auto',
            }}>
                {FIELDS.map(f => (
                    <div key={f.key} className="field-group">
                        <label htmlFor={`field-${f.key}`}>{f.label}</label>
                        {f.type === 'select' ? (
                            <select
                                id={`field-${f.key}`}
                                value={data[f.key]}
                                onChange={e => onChange(f.key, e.target.value)}
                                className={autoFilled.has(f.key) ? 'auto-filled' : ''}
                            >
                                {f.options!.map(o => (
                                    <option key={o} value={o}>{o}</option>
                                ))}
                            </select>
                        ) : (
                            <input
                                id={`field-${f.key}`}
                                type={f.type}
                                value={data[f.key]}
                                onChange={e => onChange(f.key, e.target.value)}
                                className={autoFilled.has(f.key) ? 'auto-filled' : ''}
                            />
                        )}
                    </div>
                ))}
            </div>

            {/* Submit */}
            <button
                id="btn-analyze"
                className="btn-primary"
                onClick={onSubmit}
                disabled={loading}
                style={{ width: '100%', justifyContent: 'center' }}
            >
                {loading ? '⟳  Analyzing…' : '⚡  Analyze Client'}
            </button>
        </div>
    )
}
