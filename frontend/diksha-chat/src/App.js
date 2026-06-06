import avatarImg from './assets/logo.png';
import counsellingImg from './assets/image.png';
import React, { useState, useEffect, useRef } from 'react';
import './App.css';


function renderTextWithLinks(text) {
  if (!text) return text;
  const parts = text.split(/(https?:\/\/[^\s]+)/g);
  return parts.map((part, i) => {
    if (part.match(/^https?:\/\//)) {
      return (
        <a
          key={i}
          href={part}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: '#0a66c2',
            fontWeight: 600,
            textDecoration: 'underline',
            wordBreak: 'break-all',
            display: 'inline'
          }}
        >
          🌐 {part}
        </a>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// ✅ Railway backend URL
const BACKEND_URL = 'https://motivated-forgiveness-production-1681.up.railway.app';

const LANGUAGES = [
  { code: 'en', label: '',         native: 'English' },
  { code: 'hi', label: 'Hindi',    native: 'हिंदी' },
  { code: 'ga', label: 'Garhwali', native: 'गढ़वाली' },
  { code: 'ku', label: 'Kumauni',  native: 'कुमाऊँनी' },
];

const QUICK_BUTTONS = [
  { label: 'Admissions', query: 'What is the admission process?' },
 // { label: 'Fees',       query: 'What are the fees?' },
  { label: 'Hostel',     query: 'How many Hostels are available?' },
  { label: 'Courses',    query: 'What courses are available?' },
  { label: 'Placements', query: 'What is the placement record?' },
  { label: 'Contact',    query: 'What is the contact number?' },
  { label: '🆕 New Admission', query: 'counselling2026' },
];

const getTimeGreeting = () => {
  const hour = new Date().getHours();
  if (hour >= 0  && hour < 12) return 'Good Morning';
  if (hour >= 12 && hour < 16) return 'Good Afternoon';
  return 'Good Evening';
};

const getWelcomeMessage = (langCode) => {
  const greeting = getTimeGreeting();
  const msgs = {
    en: `${greeting}! I am your friend Diksha. How may I help you?`,
    hi: `नमस्ते 🙏! मैं आपकी दोस्त दीक्षा क्या सहायता कर सकती हूँ।।`,
    ga: `समन्या 🙏! हम लग्यां छां......`,
    ku: `नमस्कार 🙏! हम काम करनी......`
  };
  return msgs[langCode];
};


// ── Diksha Avatar ─────────────────────────────────────────────────────────────
function DikshaAvatar({ speaking, size = 'small' }) {
  const dim = size === 'big' ? 52 : 40;
  return (
    <div style={{
      width: dim, height: dim,
      borderRadius: '50%',
      border: `${size === 'big' ? 3 : 2.5}px solid #C8A951`,
      overflow: 'hidden',
      flexShrink: 0,
      position: 'relative',
      background: 'transparent',
      boxShadow: '0 2px 10px rgba(0,53,128,0.35)'
    }}>
      <img
        src={avatarImg}
        alt="Diksha"
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          objectPosition: 'center 5%',
          display: 'block',
          imageRendering: '-webkit-optimize-contrast',
        }}
      />
      {speaking && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          display: 'flex', justifyContent: 'center', gap: 3,
          padding: '3px 0', background: 'rgba(0,32,96,0.75)'
        }}>
          {[0, 0.2, 0.4].map((d, i) => (
            <div key={i} style={{
              width: 4, height: 4, borderRadius: '50%',
              background: '#C8A951',
              animation: `dikshaSpeak 0.7s ${d}s infinite`
            }}/>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Course Dropdown ───────────────────────────────────────────────────────────
function CourseDropdown({ lang }) {
  const [openUG, setOpenUG] = useState(false);
  const [openPG, setOpenPG] = useState(false);

  const ugCourses = [
    { name: 'B.Tech CSE',           seats: 'Intake:60', years: '4 yr' },
    { name: 'B.Tech CSE (AI & ML)', seats: 'Intake:60', years: '4 yr' },
    { name: 'B.Tech ECE',           seats: 'Intake:60', years: '4 yr' },
    { name: 'B.Tech EE',            seats: 'Intake:60', years: '4 yr' },
    { name: 'B.Tech ME',            seats: 'Intake:60', years: '4 yr' },
    { name: 'B.Tech Civil',         seats: 'Intake:60', years: '4 yr' },
    { name: 'B.Tech Biotech',       seats: 'Intake:60', years: '4 yr' },
  ];
  const pgCourses = [
    { name: 'MCA',                    seats: 'Intake:60', years: '2 yr' },
    { name: 'M.Tech CSE',             seats: 'Intake:18', years: '2 yr' },
    { name: 'M.Tech Biotechnology',   seats: 'Intake:25', years: '2 yr' },
    { name: 'M.Tech Production Engg', seats: 'Intake:18', years: '2 yr' },
    { name: 'M.Tech Thermal Engg',    seats: 'Intake:18', years: '2 yr' },
  ];

  return (
    <div className="course-dropdown">
      <p className="dropdown-title">
        {lang === 'hi' ? ' GBPIET के कोर्स:' : ' GBPIET Courses:'}
      </p>
      <button className="dropdown-header" onClick={() => setOpenUG(!openUG)}>
        <span> {lang === 'hi' ? 'स्नातक (B.Tech)' : 'Undergraduate (B.Tech)'}</span>
        <span>{openUG ? '▲' : '▼'}</span>
      </button>
      {openUG && (
        <div className="dropdown-content">
          {ugCourses.map((c, i) => (
            <div key={i} className="course-card">
              <div className="course-name">{c.name}</div>
              <div className="course-info">
                <span> {c.seats}</span>
                <span> {c.years}</span>
              </div>
            </div>
          ))}
        </div>
      )}
      <button className="dropdown-header" onClick={() => setOpenPG(!openPG)}>
        <span> {lang === 'hi' ? 'स्नातकोत्तर' : 'Postgraduate (M.Tech/MCA)'}</span>
        <span>{openPG ? '▲' : '▼'}</span>
      </button>
      {openPG && (
        <div className="dropdown-content">
          {pgCourses.map((c, i) => (
            <div key={i} className="course-card">
              <div className="course-name">{c.name}</div>
              <div className="course-info">
                <span> {c.seats}</span>
                <span>{c.years}</span>
              </div>
            </div>
          ))}
        </div>
      )}
      <div
        className="more-link"
        onClick={() => window.open('https://gbpiet.ac.in/academic-programmes/', '_blank')}
      >
        🌐 {lang === 'hi' ? 'पूरी जानकारी देखें →' : 'View full details →'}
      </div>
    </div>
  );
}

// ── Hostel Dropdown ───────────────────────────────────────────────────────────
function HostelDropdown({ lang }) {
  const [openBoys, setOpenBoys]   = useState(false);
  const [openGirls, setOpenGirls] = useState(false);

  const boysHostels = [
    { name: 'Neelkanth Hostel', seats: 150, year: '' },
    { name: 'Kedar ABC Hostel', seats: 198, year: '' },
    { name: 'Kailash Hostel',   seats: 207, year: 'First Year' },
    { name: 'Rudra Hostel',     seats: 168, year: '' },
    { name: 'Badri Hostel',     seats: 120, year: '' },
    { name: 'Alaknanda Hostel', seats: 62,  year: '' },
    { name: 'Shivalik Hostel',  seats: 159, year: '' },
    { name: 'Trishul Hostel',   seats: 108, year: 'First Year' },
  ];

  const girlsHostels = [
    { name: 'Raman Hostel',       seats: 160, year: '' },
    { name: 'Bhagirathi Hostel',  seats: 112, year: '' },
    { name: 'Viswerwarya Hostel', seats: 144, year: 'First Year' },
  ];

  const isHi = lang === 'hi';

  const hostelCard = (h, i) => (
    <div key={i} style={{
      background: '#f5f6fa', border: '1px solid #dde3f0',
      borderLeft: '3px solid #003580', borderRadius: 8,
      padding: '8px 12px', display: 'flex',
      justifyContent: 'space-between', alignItems: 'center'
    }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#003580' }}>{h.name}</div>
        {h.year && (
          <div style={{
            fontSize: 10, background: '#E6F1FB', color: '#185FA5',
            padding: '1px 7px', borderRadius: 10, display: 'inline-block',
            marginTop: 3, fontWeight: 600
          }}>
            {isHi ? 'प्रथम वर्ष' : 'First Year'}
          </div>
        )}
      </div>
      <div style={{
        fontSize: 12, fontWeight: 700, color: '#003580',
        background: 'white', padding: '4px 10px',
        borderRadius: 20, border: '1px solid #dde3f0'
      }}>
        {h.seats}
      </div>
    </div>
  );

  const groupBtnStyle = {
    width: '100%', padding: '10px 14px',
    background: '#003580', border: 'none', color: 'white',
    borderRadius: 8, fontSize: 13, fontWeight: 600,
    cursor: 'pointer', display: 'flex',
    justifyContent: 'space-between', alignItems: 'center'
  };

  return (
    <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <p style={{ fontSize: 13, fontWeight: 700, color: '#003580', marginBottom: 2 }}>
        {isHi ? 'GBPIET छात्रावास:' : ' GBPIET Hostels:'}
      </p>

      <button style={groupBtnStyle} onClick={() => setOpenBoys(!openBoys)}>
        <span> {isHi ? 'लड़कों के हॉस्टल (8) — 1172 सीटें' : 'Boys Hostels (8) — 1172 seats'}</span>
        <span>{openBoys ? '▲' : '▼'}</span>
      </button>
      {openBoys && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5, padding: '4px 2px' }}>
          {boysHostels.map(hostelCard)}
        </div>
      )}

      <button style={groupBtnStyle} onClick={() => setOpenGirls(!openGirls)}>
        <span> {isHi ? 'लड़कियों के हॉस्टल (3) — 416 सीटें' : 'Girls Hostels (3) — 416 seats'}</span>
        <span>{openGirls ? '▲' : '▼'}</span>
      </button>
      {openGirls && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5, padding: '4px 2px' }}>
          {girlsHostels.map(hostelCard)}
        </div>
      )}

      <div
        style={{ fontSize: 11, color: '#003580', fontWeight: 600, cursor: 'pointer', marginTop: 2 }}
        onClick={() => window.open('https://gbpiet.ac.in/hostels/', '_blank')}
      >
        🌐 {isHi ? 'पूरी जानकारी देखें →' : 'View hostel details →'}
      </div>
    </div>
  );
}

// ── Admission Dropdown ────────────────────────────────────────────────────────
function AdmissionDropdown({ lang, onSelect }) {
  const programs = [
    { key: 'btech', label: 'B.Tech',  sub: 'Via JEE Main',  query: 'What is the admission process for B.Tech?' },
    { key: 'mca',   label: 'MCA',     sub: 'Via VMSBUTU',   query: 'What is the admission process for MCA?' },
    { key: 'mtech', label: 'M.Tech',  sub: 'Via GATE',      query: 'What is the admission process for M.Tech?' },
    { key: 'phd',   label: 'PhD',     sub: 'Written Exam',  query: 'What is the admission process for PhD?' },
  ];

  return (
    <div style={{ marginTop: 10 }}>
      <p style={{ fontWeight: 600 }}>
        {lang === 'hi' ? 'कौन सा कोर्स चुनना चाहते हैं?' : 'Select program:'}
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {programs.map(p => (
          <button
            key={p.key}
            onClick={() => onSelect(p.query, p.label)}
            style={{
              padding: 10, background: '#003580', color: 'white',
              border: 'none', borderRadius: 8, cursor: 'pointer'
            }}
          >
            <div>{p.label}</div>
            <small>{p.sub}</small>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Fees Dropdown ─────────────────────────────────────────────────────────────
function FeesDropdown({ lang }) {
  const [openBtech, setOpenBtech] = useState(false);
  const [openMca,   setOpenMca]   = useState(false);
  const [openMtech, setOpenMtech] = useState(false);

  const isHi = lang === 'hi';
  const MESS_FEE = 16000;

  const btechFees = [
    { sem: 'Sem 1', fee: 37980, hostel: 2480 },
    { sem: 'Sem 2', fee: 31035, hostel: 1980 },
    { sem: 'Sem 3', fee: 32280, hostel: 2480 },
    { sem: 'Sem 4', fee: 31035, hostel: 1980 },
    { sem: 'Sem 5', fee: 32280, hostel: 2480 },
    { sem: 'Sem 6', fee: 31035, hostel: 1980 },
    { sem: 'Sem 7', fee: 32280, hostel: 2480 },
    { sem: 'Sem 8', fee: 31535, hostel: 1980 },
  ];
  const mtechFees = [
    { sem: 'Sem 1', fee: 43480, hostel: 2480 },
    { sem: 'Sem 2', fee: 36535, hostel: 1980 },
    { sem: 'Sem 3', fee: 37780, hostel: 2480 },
    { sem: 'Sem 4', fee: 36835, hostel: 1980 },
  ];
  const mcaFees = [
    { sem: 'Sem 1', fee: 43480, hostel: 2480 },
    { sem: 'Sem 2', fee: 36535, hostel: 1980 },
    { sem: 'Sem 3', fee: 37780, hostel: 2480 },
    { sem: 'Sem 4', fee: 36835, hostel: 1980 },
  ];

  const renderFees = (data) => (
    <div style={{ padding: '6px 10px' }}>
      {data.map((f, i) => {
        const total = f.fee + f.hostel + MESS_FEE;
        return (
          <div key={i} style={{
            background: '#f5f6fa', borderLeft: '3px solid #003580',
            padding: '8px 12px', marginBottom: 6, borderRadius: 8
          }}>
            <div style={{ fontWeight: 600, color: '#003580' }}>{f.sem}</div>
            <div style={{ fontSize: 12 }}> Institute: ₹{f.fee}</div>
            <div style={{ fontSize: 12 }}> Mess: ₹{MESS_FEE}</div>
            <div style={{ fontSize: 12 }}> Hostel: ₹{f.hostel}</div>
            <div style={{ marginTop: 4, fontWeight: 700, color: '#0a7f3f' }}>
               Total: ₹{total}
            </div>
          </div>
        );
      })}
    </div>
  );

  return (
    <div style={{ marginTop: 10 }}>
      <p style={{ fontWeight: 600 }}>
        {isHi ? ' फीस संरचना (सेमेस्टर वाइज)' : ' Fees Structure (Semester-wise)'}
      </p>
      <button className="dropdown-header" onClick={() => setOpenBtech(!openBtech)}>
        B.Tech {openBtech ? '▲' : '▼'}
      </button>
      {openBtech && renderFees(btechFees)}

      <button className="dropdown-header" onClick={() => setOpenMca(!openMca)}>
        MCA {openMca ? '▲' : '▼'}
      </button>
      {openMca && renderFees(mcaFees)}

      <button className="dropdown-header" onClick={() => setOpenMtech(!openMtech)}>
        M.Tech {openMtech ? '▲' : '▼'}
      </button>
      {openMtech && renderFees(mtechFees)}

      <div style={{ marginTop: 8, fontSize: 11 }}>
        🌐 Pay Fees:
        <br />
        <a href="https://onlinesbi.sbi.bank.in/sbicollect/icollecthome.htm?corpID=823332" target="_blank" rel="noopener noreferrer">
          Institute Fee
        </a>
        <br />
        <a href="https://onlinesbi.sbi.bank.in/sbicollect/icollecthome.htm?corpID=908435" target="_blank" rel="noopener noreferrer">
          Hostel &amp; Mess Fee
        </a>
      </div>
    </div>
  );
}


// ══ MAIN APP ══════════════════════════════════════════════════════════════════
export default function App() {
  const [stage, setStage]             = useState('welcome');
  const [messages, setMessages]       = useState([]);
  const [input, setInput]             = useState('');
  const [loading, setLoading]         = useState(false);
  const [sessionId, setSessionId]     = useState(null);
  const [language, setLanguage]       = useState(null);
  const [isSpeaking, setIsSpeaking]   = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [currentLang, setCurrentLang] = useState('en');
  const [backendReady, setBackendReady] = useState(false);  // ✅ NEW
  const messagesEndRef = useRef(null);
  const audioRef       = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ✅ Single sessionId effect
  useEffect(() => {
    let sid = localStorage.getItem('session_id');
    if (!sid) {
      sid = crypto.randomUUID();
      localStorage.setItem('session_id', sid);
    }
    setSessionId(sid);
  }, []);

  // ✅ Wake up Railway backend — single effect, sets backendReady
  useEffect(() => {
    const wakeUp = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/health`);
        const data = await res.json();
        console.log('[Diksha] Backend awake:', data.status);
        setBackendReady(true);
      } catch (err) {
        console.warn('[Diksha] Wake-up failed:', err.message);
        setBackendReady(true); // allow anyway so UI isn't stuck
      }
    };
    wakeUp();
  }, []);

  // ✅ Stop audio when tab is hidden
  useEffect(() => {
    const handleVisibility = () => { if (document.hidden) stopSpeaking(); };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, []);


  // ── TTS ───────────────────────────────────────────────────────────────────
  const playAudio = (base64Audio, onEnd) => {
    try {
      if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
      if (!base64Audio) { if (onEnd) onEnd(); return; }
      const audio = new Audio(`data:audio/mp3;base64,${base64Audio}`);
      audioRef.current = audio;
      audio.onplay  = () => setIsSpeaking(true);
      audio.onended = () => { setIsSpeaking(false); audioRef.current = null; if (onEnd) onEnd(); };
      audio.onerror = () => { setIsSpeaking(false); audioRef.current = null; if (onEnd) onEnd(); };
      audio.play();
    } catch (e) {
      console.log('Audio error:', e);
      setIsSpeaking(false);
    }
  };

  const expandTextForTTS = (text, lang = 'en') => {
    if (lang === 'en') {
      return text
        .replace(/\bDr\./gi, 'Doctor')
        .replace(/\bDr\b/gi, 'Doctor')
        .replace(/\bProf\./gi, 'Professor')
        .replace(/\bProf\b/gi, 'Professor')
        .replace(/\bHOD\b/gi, 'H O D')
        .replace(/\bGBPIET\b/gi, 'G B P I E T');
    }
    if (lang === 'hi') {
      return text
        .replace(/\bDr\./gi, 'डॉक्टर')
        .replace(/\bDr\b/gi, 'डॉक्टर')
        .replace(/\bProf\./gi, 'प्रोफेसर')
        .replace(/\bProf\b/gi, 'प्रोफेसर')
        .replace(/प्रो\./g, 'प्रोफेसर')
        .replace(/डॉ\./g, 'डॉक्टर')
        .replace(/जीबीपीआईईटी/g, 'जी बी पी आई ई टी')
        .replace(/\bGBPIET\b/gi, 'G B P I E T')
        .replace(/हैं/g, 'हैं ')
        .replace(/है/g, 'है ');
    }
    return text;
  };

  const fetchAndPlayTTS = async (text, lang, onEnd) => {
    try {
      const cleanText = expandTextForTTS(text, lang);
      const res = await fetch(`${BACKEND_URL}/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: cleanText, lang })
      });
      const data = await res.json();
      if (data.audio_base64) playAudio(data.audio_base64, onEnd);
      else if (onEnd) onEnd();
    } catch (e) {
      console.log('TTS error:', e);
      if (onEnd) onEnd();
    }
  };

  const stopSpeaking = () => {
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
    setIsSpeaking(false);
  };


  // ── Language handlers ──────────────────────────────────────────────────────
  const handleLangSelect = async (langCode) => {
    setLanguage(langCode);
    setCurrentLang(langCode);
    const welcomeText = getWelcomeMessage(langCode);
    setMessages([{
      role: 'diksha', text: welcomeText, lang: langCode,
      time: new Date().toLocaleTimeString()
    }]);
    setStage('chat');
    await fetchAndPlayTTS(welcomeText, langCode);
  };

  const handleLangSwitch = async (langCode) => {
    if (langCode === currentLang) return;
    stopSpeaking();
    setCurrentLang(langCode);
    setLanguage(langCode);
    setInput('');
    setLoading(false);
    const welcomeText = getWelcomeMessage(langCode);
    setMessages([{
      role: 'diksha', text: welcomeText, lang: langCode,
      time: new Date().toLocaleTimeString()
    }]);
    await fetchAndPlayTTS(welcomeText, langCode);
  };


  // ── Intent detectors ───────────────────────────────────────────────────────
  const isFeesQuery = (q) => {
    const lower = q.toLowerCase().trim();
    return ['fees','fee structure','btech fees','mca fees','mtech fees',
      'college fees','how much fee','kitni fees','fees kitni hai',
      'fee batao','फीस','शुल्क'].some(k => lower.includes(k));
  };

  const isCourseQuery = (q) => {
    const lower = q.toLowerCase().trim();
    const isAdmission = ['admission','process','apply','how to','eligibility',
      'jee','gate','utuee','document','seat','प्रवेश','दाखिला','आवेदन']
      .some(w => lower.includes(w));
    if (isAdmission) return false;
    return ['what courses','which courses','list of courses','courses available',
      'courses offered','all courses','available courses','course list',
      'programs offered','courses at gbpiet','branches at gbpiet','what branch',
      'which branch','all branches','कोर्स','शाखा','कार्यक्रम',
      'सभी कोर्स','कौन से कोर्स'].some(k => lower.includes(k));
  };

  const isAdmissionQuery = (q) => {
    const lower = q.toLowerCase().trim();
    return ['admission process','admission','how to apply','how to get admission',
      'प्रवेश','दाखिला'].some(k => lower.includes(k));
  };

  const isHostelQuery = (q) => {
    const lower = q.toLowerCase().trim();
    return ['how many hostel','list of hostel','hostel list','hostel name',
      'all hostel','boys hostel','girls hostel','hostel available',
      'available hostel','hostel at gbpiet','hostel facility','hostel details',
      'hostel information','कितने हॉस्टल','हॉस्टल की सूची','सभी हॉस्टल',
      'हॉस्टल की जानकारी','hostel mein','hostel hai','how many hostels']
      .some(k => lower.includes(k));
  };


  // ── Admission select handler ───────────────────────────────────────────────
  const handleAdmissionSelect = (query, label) => {
    setMessages(prev => [...prev, {
      role: 'user', text: label, time: new Date().toLocaleTimeString()
    }]);
    handleSend(query, true);
  };


  // ── Main send handler ──────────────────────────────────────────────────────
  const handleSend = async (questionText, skipIntent = false) => {
    const q = (questionText || input).trim();
    if (!q || !language) return;

    setMessages(prev => [...prev, {
      role: 'user', text: q, time: new Date().toLocaleTimeString()
    }]);
    setInput('');
    setLoading(true);
// ── New Admission / Counselling 2026 ──────────────────────────────
if (q === 'counselling2026') {
  setMessages(prev => [...prev, {
    role: 'diksha',
    text: language === 'hi'
      ? 'UKTECH ऑनलाइन काउंसलिंग 2026-27 की सूचना:'
      : 'UKTECH Online Counselling 2026-27 Notice:',
    lang: language,
    type: 'counselling2026',
    time: new Date().toLocaleTimeString()
  }]);
  setLoading(false);
  return;
}
    // ── Local intent handlers (no backend needed) ──────────────────────────
    if (!skipIntent && isCourseQuery(q)) {
      const txt = language === 'hi'
        ? 'यहाँ GBPIET के सभी कोर्स की जानकारी है:'
        : 'Here are all courses offered at GBPIET:';
      setMessages(prev => [...prev, {
        role: 'diksha', text: txt, lang: language,
        type: 'courses', time: new Date().toLocaleTimeString()
      }]);
      setLoading(false);
      return;
    }

    if (!skipIntent && isAdmissionQuery(q)) {
      const txt = language === 'hi'
        ? 'किस कोर्स की प्रवेश प्रक्रिया जानना चाहते हैं?'
        : 'Which program admission process do you want?';
      setMessages(prev => [...prev, {
        role: 'diksha', text: txt, lang: language,
        type: 'admission', time: new Date().toLocaleTimeString()
      }]);
      setLoading(false);
      return;
    }

    if (!skipIntent && isHostelQuery(q)) {
      const txt = language === 'hi'
        ? 'यहाँ GBPIET के सभी छात्रावासों की जानकारी है:'
        : 'Here are all hostels at GBPIET:';
      setMessages(prev => [...prev, {
        role: 'diksha', text: txt, lang: language,
        type: 'hostels', time: new Date().toLocaleTimeString()
      }]);
      setLoading(false);
      return;
    }

    if (!skipIntent && isFeesQuery(q)) {
      const txt = language === 'hi'
        ? 'यहाँ GBPIET की फीस संरचना है:'
        : 'Here is the fee structure of GBPIET:';
      setMessages(prev => [...prev, {
        role: 'diksha', text: txt, lang: language,
        type: 'fees', time: new Date().toLocaleTimeString()
      }]);
      setLoading(false);
      return;
    }

    // ── Backend call ───────────────────────────────────────────────────────
    try {
      const controller = new AbortController();
      const timeoutId  = setTimeout(() => controller.abort(), 90000); // ✅ 90s timeout

      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question:         q,
          session_id:       sessionId,
          is_first_message: false,
          language
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      if (!sessionId) setSessionId(data.session_id);

      setMessages(prev => [...prev, {
        role: 'diksha', text: data.answer,
        lang: language, time: new Date().toLocaleTimeString()
      }]);

    } catch (err) {
      console.error('[Diksha] Fetch error →', err.name, ':', err.message);

      const isTimeout = err.name === 'AbortError';

      const errorMsg = language === 'hi'
        ? isTimeout
          ? 'सर्वर जवाब देने में समय लग रहा है। कृपया 10 सेकंड बाद दोबारा कोशिश करें।'
          : 'माफ़ करें, सर्वर से जुड़ नहीं पा रहे हैं। कृपया दोबारा कोशिश करें।'
        : isTimeout
          ? 'Server is taking too long to respond. Please try again in 10 seconds.'
          : 'Sorry, unable to connect to server. Please try again.';

      setMessages(prev => [...prev, {
        role: 'diksha', text: errorMsg,
        lang: language, time: new Date().toLocaleTimeString()
      }]);
    }

    setLoading(false);
  };


  // ── Voice input ────────────────────────────────────────────────────────────
  const startListening = () => {
    stopSpeaking();
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { alert('Please use Chrome for voice input!'); return; }
    const r = new SR();
    r.lang      = language === 'en' ? 'en-IN' : 'hi-IN';
    r.onstart   = () => setIsListening(true);
    r.onend     = () => setIsListening(false);
    r.onresult  = (e) => { const t = e.results[0][0].transcript; setInput(t); handleSend(t); };
    r.onerror   = () => setIsListening(false);
    r.start();
  };


  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      {/* ══ 1. FLOATING FAB ══ */}
      {stage !== 'chat' && window.self === window.top && (
        <div className="floating-fab" onClick={() => setStage('welcome')}>
          <div className="fab-pulse-ring"/>
          <div className="fab-circle">
            <img
              src={avatarImg}
              alt="Diksha"
              style={{
                width: '100%', height: '100%',
                objectFit: 'cover', objectPosition: 'center 5%',
                display: 'block', imageRendering: '-webkit-optimize-contrast'
              }}
            />
          </div>
          <div className="fab-label">WELCOME GBPIET</div>
          <div className="fab-online-dot"/>
        </div>
      )}

      {/* ══ 2. LANGUAGE POPUP ══ */}
      {stage === 'welcome' && (
        <div
          className="popup-overlay"
          onClick={e => { if (e.target.classList.contains('popup-overlay')) setStage('Welcome'); }}
        >
          <div className="popup-box">
            <div className="popup-header">
              <img
                src="https://gbpiet.ac.in/wp-content/uploads/2023/03/logo-final.png"
                alt="GBPIET"
                className="popup-logo"
                onError={e => (e.target.style.display = 'none')}
              />
              <div className="popup-header-text">
                <div className="popup-college-name">
                  गोविंद बल्लभ पंत अभियान्त्रिकी एवं प्रौद्योगिकी संस्थान
                </div>
                <div className="popup-college-en">
                  Govind Ballabh Pant Institute of Engineering &amp; Technology
                </div>
                <div className="popup-college-sub">
                  Pauri Garhwal, Uttarakhand — An Autonomous Institute of Govt. of Uttarakhand
                </div>
              </div>
              <button
                className="drawer-icon-btn"
                onClick={() => {
                  stopSpeaking();
                  setStage('idle');
                  window.parent.postMessage('CLOSE_DIKSHA', '*');
                }}
              >✕</button>
            </div>

            <div className="popup-diksha-row">
             <div style={{
  width:72, height:72,
  borderRadius:'0',
  border:'none',
  overflow:'visible',
  background:'none',
  flexShrink:0,
  boxShadow:'none'
}}>
              <img src={avatarImg} alt="Diksha" style={{
  width: '100%',
  height: '100%',
  objectFit: 'contain',
  objectPosition: 'center',
  display: 'block',
  background: 'none',
  imageRendering: '-webkit-optimize-contrast'
}}/>
              </div>
              <div className="popup-intro-text">
                <p className="popup-greeting">{getTimeGreeting()}!</p>
                <p className="popup-hi">GBPIET में आपका स्वागत है | Welcome to GBPIET</p>
                <p className="popup-sub">अपनी भाषा चुनें | Select your Preferred Language</p>
              </div>
            </div>

            <div className="popup-lang-grid">
              {LANGUAGES.map(l => (
                <button
                  key={l.code}
                  className="popup-lang-btn"
                  onClick={() => handleLangSelect(l.code)}
                >
                  <span className="plb-flag">{l.flag}</span>
                  <span className="plb-native">{l.native}</span>
                  <span className="plb-en">{l.label}</span>
                </button>
              ))}
            </div>

            <p className="popup-footer">
               &nbsp;•&nbsp; UGC Autonomous &nbsp;•&nbsp; AICTE &nbsp;•&nbsp; UHV &nbsp;•&nbsp; IKS &nbsp;•&nbsp;
            </p>
          </div>
        </div>
      )}

      {/* ══ 3. CHAT DRAWER ══ */}
      {stage === 'chat' && (
        <div className="chat-drawer">

          {/* Header */}
          <div className="drawer-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <DikshaAvatar speaking={isSpeaking} size="big"/>
              <div>
                <div className="drawer-name">Diksha — दीक्षा</div>
                {/* ✅ Shows connecting status until backend is ready */}
                <div className="drawer-status">
                  {!backendReady     ? '⏳ Connecting to server...'
                  : isSpeaking       ? '🔊 Speaking...'
                  : isListening      ? '🎤 Listening...'
                  : '● GBPIET Collegemate-your tour guide'}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {isSpeaking && (
                <button className="drawer-icon-btn" onClick={stopSpeaking}>🔇</button>
              )}
              <button
                className="drawer-icon-btn"
                onClick={() => { stopSpeaking(); setStage('Welcome'); }}
              >✕</button>
            </div>
          </div>

          {/* Language pills */}
          <div className="drawer-lang-row">
            {LANGUAGES.map(l => (
              <button
                key={l.code}
                className={`lang-pill ${currentLang === l.code ? 'active' : ''}`}
                onClick={() => handleLangSwitch(l.code)}
                title={`Switch to ${l.label}`}
              >
                {l.flag} {l.native}
              </button>
            ))}
          </div>

          {/* Messages */}
          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-row ${msg.role}`}>
                {msg.role === 'diksha' && (
                  <DikshaAvatar
                    speaking={isSpeaking && idx === messages.length - 1}
                    size="small"
                  />
                )}
                <div className="msg-content">
                  {msg.role === 'diksha' && (
                    <span className="msg-sender-name">Diksha</span>
                  )}
                  <div className={`msg-bubble ${msg.isLangSwitch ? 'lang-switch-bubble' : ''}`}>
                    {msg.isLangSwitch && <span className="lang-switch-icon">🌐 </span>}

                    <div>{renderTextWithLinks(msg.text)}</div>

                    {msg.role === 'diksha' && msg.text.toLowerCase().includes('placement') && (
                      <div style={{ marginTop: 8 }}>
                        <a
                          href="https://gbpiet.ac.in/placement-records/"
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ color: '#0a66c2', fontWeight: '600', textDecoration: 'none' }}
                        >
                          🌐 View Full Placement Details →
                        </a>
                      </div>
                    )}

                    {msg.type === 'courses'   && <CourseDropdown lang={msg.lang}/>}
                    {msg.type === 'hostels'   && <HostelDropdown lang={msg.lang}/>}
                    {msg.type === 'admission' && (
                      <AdmissionDropdown lang={msg.lang} onSelect={handleAdmissionSelect}/>
                    )}
                    {msg.type === 'fees' && <FeesDropdown lang={msg.lang}/>}
                   
                    {msg.type === 'counselling2026' && (
                      <div style={{ marginTop: 8 }}>
                        <img
                          src={counsellingImg}
                          alt="UKTECH Counselling 2026-27"
                          style={{
                            width: '100%',
                            borderRadius: 8,
                            border: '2px solid #8B0000',
                            cursor: 'pointer',
                          }}
                          onClick={() => window.open('https://uktech.ac.in', '_blank')}
                        />
                        <div style={{
                          fontSize: 11,
                          color: '#8B0000',
                          fontWeight: 600,
                          marginTop: 6,
                          textAlign: 'center',
                        }}>
                          🌐 {language === 'hi'
                            ? 'पूरी जानकारी के लिए uktech.ac.in पर जाएं →'
                            : 'Click image to visit uktech.ac.in →'}
                        </div>
                        <div style={{
                          marginTop: 8,
                          fontSize: 12,
                          background: '#fff8e1',
                          border: '1px solid #f0c040',
                          borderRadius: 6,
                          padding: '8px 10px',
                          color: '#5a3e00',
                          lineHeight: 1.8,
                        }}>
                          📅 {language === 'hi' ? 'पहला चरण: 28 मई – 10 जून 2026'    : 'Phase 1: 28 May – 10 Jun 2026'}<br/>
                          📅 {language === 'hi' ? 'दूसरा चरण: 12 – 24 जून 2026'       : 'Phase 2: 12 – 24 Jun 2026'}<br/>
                          📅 {language === 'hi' ? 'तीसरा चरण: 26 जून – 09 जुलाई 2026' : 'Phase 3: 26 Jun – 09 Jul 2026'}
                        </div>
                      </div>
                    )}

                    
                  </div>
                      

                  <div className="msg-meta">
                    <span className="msg-time">{msg.time}</span>
                    {msg.role === 'diksha' && (
                      <button
                        className="speak-btn"
                        title="Click to hear Diksha"
                        onClick={() => {
                          if (isSpeaking) stopSpeaking();
                          else fetchAndPlayTTS(msg.text, msg.lang);
                        }}
                      >
                        {isSpeaking ? '🔇' : '🔊'}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-row diksha">
                <DikshaAvatar speaking={false} size="small"/>
                <div className="msg-content">
                  <span className="msg-sender-name">Diksha</span>
                  <div className="msg-bubble typing">
                    <span/><span/><span/>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef}/>
          </div>

          {/* Quick buttons */}
          <div className="quick-buttons">
            {QUICK_BUTTONS.map((btn, i) => (
              <button
                key={i}
                className="quick-btn"
                onClick={() => handleSend(btn.query)}
              >
                {btn.label}
              </button>
            ))}
          </div>

          {/* Input */}
          <div className="chat-input-area">
            <div className="input-row">
              <button
                className={`mic-btn ${isListening ? 'active' : ''}`}
                onClick={startListening}
              >
                {isListening ? '🔴' : '🎤'}
              </button>
              <textarea
                value={input}
                onChange={e => { stopSpeaking(); setInput(e.target.value); }}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={language === 'hi'
                  ? 'GBPIET के बारे में कुछ भी पूछें...'
                  : 'Ask anything about GBPIET...'}
                className="chat-input"
                rows={1}
                disabled={!language || loading}
              />
              {input && (
                <button onClick={() => setInput('')} title="Clear">✕</button>
              )}
              {/* ✅ Disabled until backend is ready */}
              <button
                className="send-btn"
                onClick={() => handleSend()}
                disabled={loading || !language || !input.trim() || !backendReady}
              >
                {loading ? '⏳' : '➤'}
              </button>
            </div>
            <p className="input-hint">
              {!backendReady ? '⏳ Waiting for server...' : 'Press Enter to send'}
            </p>
          </div>

          <div className="drawer-disclaimer">
            Beta V26.2 (06/05/26) &nbsp;|&nbsp; Team MCA Supervised by KDN
          </div>
        </div>
      )}
    </>
  );
}
