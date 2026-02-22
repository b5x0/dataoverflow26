import { useRef, useState, useEffect, useCallback } from 'react'
import { Mic, MicOff, Zap, Radio } from 'lucide-react'
import VoicePulse from './components/VoicePulse'
import LiveTranscription, { type TranscriptEntry } from './components/LiveTranscription'
import ImpactDriverChart, { type ShapEntry } from './components/ImpactDriverChart'
import LookalikeCard, { type Lookalike } from './components/LookalikeCard'
import SalesHUD from './components/SalesHUD'
import ManualOverrideForm, { type FormData } from './components/ManualOverrideForm'

// ── Constants ──────────────────────────────────────────────────
const WS_URL = 'ws://localhost:8002/ws/advisor'
const REST_URL = 'http://localhost:8002/predict'
const SAMPLE_RATE = 16000

const DEFAULT_FORM: FormData = {
    Estimated_Annual_Income: '50000',
    Employment_Status: 'Full-Time',
    Region_Code: 'Unknown',
    Adult_Dependents: '0',
    Child_Dependents: '0',
    Infant_Dependents: '0',
    Previous_Claims_Filed: '0',
    Years_Without_Claims: '1',
    Deductible_Tier: 'Medium',
    Acquisition_Channel: 'Online',
    Payment_Schedule: 'Monthly',
    Policy_Start_Month: 'January',
    Existing_Policyholder: '0',
    Previous_Policy_Duration_Months: '12',
    Vehicles_on_Policy: '1',
    Custom_Riders_Requested: '0',
    Grace_Period_Extensions: '0',
    Policy_Cancelled_Post_Purchase: '0',
}

interface AnalysisResult {
    prediction: number
    confidence: number
    shap: ShapEntry[]
    lookalikes: Lookalike[]
    extracted_form: Record<string, unknown>
    predict_ms: number
}

// ── PCM float32 → int16 bytes ──────────────────────────────────
function floatToInt16(float32Array: Float32Array): ArrayBuffer {
    const buf = new ArrayBuffer(float32Array.length * 2)
    const view = new DataView(buf)
    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]))
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    }
    return buf
}

// ── App ────────────────────────────────────────────────────────
export default function App() {
    // ── Session state
    const [sessionActive, setSessionActive] = useState(false)
    const [audioLevel, setAudioLevel] = useState(0)
    const [speaking, setSpeaking] = useState(false)  // AI speaking
    const [formLoading, setFormLoading] = useState(false)

    // ── Content state
    const [transcript, setTranscript] = useState<TranscriptEntry[]>([])
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
    const [salesPitch, setSalesPitch] = useState<string | null>(null)
    const [formData, setFormData] = useState<FormData>(DEFAULT_FORM)
    const [autoFilled, setAutoFilled] = useState<Set<string>>(new Set())

    // ── Refs
    const wsRef = useRef<WebSocket | null>(null)
    const audioCtxRef = useRef<AudioContext | null>(null)
    const analyserRef = useRef<AnalyserNode | null>(null)
    const scriptRef = useRef<ScriptProcessorNode | null>(null)
    const streamRef = useRef<MediaStream | null>(null)
    const playQueueRef = useRef<ArrayBuffer[]>([])
    const isPlayingRef = useRef(false)
    const transcriptRef = useRef<TranscriptEntry[]>([])
    const idRef = useRef(0)
    const levelTimerRef = useRef<number>(0)
    const aiTextBufRef = useRef<string>('')
    const recognitionRef = useRef<any>(null)

    // ── Audio level polling
    const startLevelPoll = useCallback(() => {
        const analyser = analyserRef.current
        if (!analyser) return
        const buf = new Uint8Array(analyser.fftSize)
        const tick = () => {
            analyser.getByteTimeDomainData(buf)
            let sum = 0
            for (let i = 0; i < buf.length; i++) {
                const s = buf[i] - 128
                sum += s * s
            }
            setAudioLevel(Math.sqrt(sum / buf.length) * 4)
            levelTimerRef.current = requestAnimationFrame(tick)
        }
        levelTimerRef.current = requestAnimationFrame(tick)
    }, [])

    // ── Play PCM audio from Gemini (24kHz, 16-bit, mono)
    const playPCMChunk = useCallback(async (buf: ArrayBuffer) => {
        playQueueRef.current.push(buf)
        if (isPlayingRef.current) return
        isPlayingRef.current = true
        setSpeaking(true)

        const ctx = audioCtxRef.current ?? new AudioContext({ sampleRate: 24000 })
        audioCtxRef.current = ctx

        while (playQueueRef.current.length > 0) {
            const chunk = playQueueRef.current.shift()!
            const view = new DataView(chunk)
            const samples = new Float32Array(chunk.byteLength / 2)
            for (let i = 0; i < samples.length; i++) {
                samples[i] = view.getInt16(i * 2, true) / 0x8000
            }
            const audioBuf = ctx.createBuffer(1, samples.length, 24000)
            audioBuf.copyToChannel(samples, 0)
            const src = ctx.createBufferSource()
            src.buffer = audioBuf
            src.connect(ctx.destination)
            await new Promise<void>(resolve => {
                src.onended = () => resolve()
                src.start()
            })
        }

        isPlayingRef.current = false
        setSpeaking(false)
    }, [])

    // ── Start session
    const startSession = useCallback(async () => {
        // Init Speech Recognition (since Live model is audio-only text-wise)
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
        if (SpeechRecognition) {
            const rec = new SpeechRecognition()
            rec.continuous = true
            rec.interimResults = true
            rec.lang = 'en-US'
            rec.onresult = (e: any) => {
                let interim = ''
                let final = ''
                for (let i = e.resultIndex; i < e.results.length; i++) {
                    if (e.results[i].isFinal) final += e.results[i][0].transcript
                    else interim += e.results[i][0].transcript
                }
                const full = final || interim
                if (full) {
                    if (wsRef.current?.readyState === WebSocket.OPEN) {
                        wsRef.current.send(JSON.stringify({ type: 'USER_TRANSCRIPT', content: full }))
                    }
                    setTranscript(prev => {
                        const last = prev[prev.length - 1]
                        if (last?.speaker === 'agent' && last.id === 9999) {
                            const updated = [...prev]
                            updated[updated.length - 1] = { ...last, text: full }
                            return updated
                        }
                        return [...prev, { id: 9999, speaker: 'agent', text: full }]
                    })
                }
            }
            rec.start()
            recognitionRef.current = rec
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: SAMPLE_RATE,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            })
            streamRef.current = stream

            const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
            const src = ctx.createMediaStreamSource(stream)
            const analyser = ctx.createAnalyser()
            analyser.fftSize = 512
            analyserRef.current = analyser
            src.connect(analyser)

            const processor = ctx.createScriptProcessor(1024, 1, 1) // Reduced for lower lag
            scriptRef.current = processor
            processor.onaudioprocess = (e) => {
                if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
                wsRef.current.send(floatToInt16(e.inputBuffer.getChannelData(0)))
            }
            analyser.connect(processor)
            processor.connect(ctx.destination)

            startLevelPoll()

            const ws = new WebSocket(WS_URL)
            ws.binaryType = 'arraybuffer'
            wsRef.current = ws

            ws.onopen = () => {
                setSessionActive(true)
                setTranscript([])
                idRef.current = 0
                aiTextBufRef.current = ''
            }

            ws.onmessage = async (evt) => {
                if (evt.data instanceof ArrayBuffer) {
                    await playPCMChunk(evt.data)
                } else {
                    try {
                        const msg = JSON.parse(evt.data)
                        if (msg.type === 'TRANSCRIPT') {
                            aiTextBufRef.current += msg.content
                            setTranscript(prev => {
                                const last = prev[prev.length - 1]
                                if (last?.speaker === 'ai' && last.id === idRef.current) {
                                    const updated = [...prev]
                                    updated[updated.length - 1] = { ...last, text: aiTextBufRef.current }
                                    return updated
                                }
                                const entry: TranscriptEntry = { id: ++idRef.current, speaker: 'ai', text: aiTextBufRef.current }
                                return [...prev, entry]
                            })
                        } else if (msg.type === 'ANALYSIS') {
                            setAnalysis(msg as AnalysisResult)
                            if (msg.extracted_form) {
                                setFormData(prev => ({ ...prev, ...msg.extracted_form }))
                            }
                        } else if (msg.type === 'TURN_COMPLETE') {
                            aiTextBufRef.current = ''
                        }
                    } catch (e) { console.error('WS Error:', e) }
                }
            }

            ws.onclose = () => setSessionActive(false)
        } catch (err) {
            console.error('Session failed:', err)
        }
    }, [startLevelPoll, playPCMChunk])

    // ── Stop session
    const stopSession = useCallback(() => {
        cancelAnimationFrame(levelTimerRef.current)
        setAudioLevel(0)
        recognitionRef.current?.stop()
        wsRef.current?.send('STOP')
        wsRef.current?.close()
        scriptRef.current?.disconnect()
        streamRef.current?.getTracks().forEach(t => t.stop())
        setSessionActive(false)
        setSpeaking(false)
    }, [])

    useEffect(() => () => stopSession(), [stopSession])

    // ── Manual form submit
    const handleManualSubmit = useCallback(async () => {
        setFormLoading(true)
        try {
            const body: Record<string, unknown> = {}
            Object.entries(formData).forEach(([k, v]) => {
                body[k] = isNaN(Number(v)) ? v : Number(v)
            })
            const res = await fetch(REST_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            })
            const result: AnalysisResult = await res.json()
            setAnalysis(result)
        } catch (err) {
            console.error('Manual submit failed:', err)
        } finally {
            setFormLoading(false)
        }
    }, [formData])

    const handleFormChange = useCallback((field: string, value: string) => {
        setFormData(prev => ({ ...prev, [field]: value }))
        setAutoFilled(prev => { const n = new Set(prev); n.delete(field); return n })
    }, [])

    // ── Layout ─────────────────────────────────────────────────────
    return (
        <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg)' }}>
            <header style={{
                padding: '14px 28px', borderBottom: '1px solid var(--border)',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                background: 'rgba(13,27,46,0.95)', backdropFilter: 'blur(12px)', flexShrink: 0,
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                    <Radio size={22} color="var(--cyan)" />
                    <div>
                        <div style={{ fontSize: '1rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
                            Voice-QL <span style={{ color: 'var(--cyan)' }}>Spokesman HUD</span>
                        </div>
                        <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                            AI Agent calling Client · v2.1 · DataOverflow
                        </div>
                    </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    {sessionActive && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.72rem', color: speaking ? 'var(--purple)' : 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>
                            <span style={{
                                width: '6px', height: '6px', borderRadius: '50%',
                                background: speaking ? 'var(--purple)' : 'var(--green)',
                                boxShadow: `0 0 6px ${speaking ? 'var(--purple)' : 'var(--green)'}`,
                                animation: 'pulse-ring-fast 1s ease-in-out infinite',
                            }} />
                            {speaking ? 'AI AGENT SPEAKING' : 'LISTENING (CLIENT)'}
                        </div>
                    )}
                    {analysis && <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>predict: {analysis.predict_ms}ms</div>}
                    {sessionActive ? (
                        <button id="btn-stop" className="btn-danger" onClick={stopSession}><MicOff size={14} /> End Session</button>
                    ) : (
                        <button id="btn-start" className="btn-primary" onClick={startSession}><Mic size={14} /> Start Session</button>
                    )}
                </div>
            </header>

            <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '300px 1fr 340px', gap: '16px', padding: '16px 20px', overflow: 'hidden', minHeight: 0 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden' }}>
                    <div className="glass" style={{ height: '180px', position: 'relative', padding: '8px', flexShrink: 0 }}>
                        <div className="label">Voice Input</div>
                        <VoicePulse level={audioLevel} active={sessionActive} speaking={speaking} />
                    </div>
                    <div className="glass" style={{ padding: '14px', flexShrink: 0 }}>
                        <div className="label" style={{ marginBottom: '8px' }}><Zap size={10} style={{ display: 'inline', marginRight: '4px' }} />Spokesman Script</div>
                        <SalesHUD pitch={salesPitch} />
                    </div>
                    <div className="glass" style={{ flex: 1, padding: '14px', display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>
                        <div className="label" style={{ marginBottom: '8px', flexShrink: 0 }}>Live Transcription</div>
                        <LiveTranscription entries={transcript} />
                    </div>
                </div>

                <div className="glass" style={{ padding: '16px', display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>
                    <div className="label" style={{ marginBottom: '10px', flexShrink: 0 }}>Manual Override Form <span style={{ color: 'var(--cyan-dim)', fontWeight: 400, marginLeft: '6px' }}>— auto-fills from voice</span></div>
                    <ManualOverrideForm data={formData} autoFilled={autoFilled} onChange={handleFormChange} onSubmit={handleManualSubmit} loading={formLoading} />
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden' }}>
                    <div className="glass" style={{ padding: '14px', flex: 1, overflow: 'hidden', minHeight: 0, overflowY: 'auto' }}>
                        <div className="label" style={{ marginBottom: '10px' }}>Impact Drivers (SHAP)</div>
                        <ImpactDriverChart data={analysis?.shap ?? []} prediction={analysis?.prediction ?? null} confidence={analysis?.confidence ?? null} />
                    </div>
                    <div className="glass" style={{ padding: '14px', flex: 1, overflow: 'hidden', minHeight: 0, overflowY: 'auto' }}>
                        <div className="label" style={{ marginBottom: '10px' }}>Clients Like Yours</div>
                        <LookalikeCard clients={analysis?.lookalikes ?? []} />
                    </div>
                </div>
            </div>
        </div>
    )
}
