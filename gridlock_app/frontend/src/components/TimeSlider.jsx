import { useEffect, useState } from 'react';
import { Play, Pause, SkipBack, SkipForward, Clock, FastForward } from 'lucide-react';

export const TimeSlider = ({ slotIndex, onSlotChange }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1); // 1x, 2x, 4x

  // Compute time string from slotIndex (0 = 00:00, 95 = 23:45)
  const totalMinutes = slotIndex * 15;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const displayHours = hours % 12 === 0 ? 12 : hours % 12;
  const timeString = `${displayHours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')} ${ampm}`;

  // Peak categorization
  let periodTag = 'Off-Peak / Night';
  let periodColor = 'text-slate-400 bg-slate-800/60';
  if (hours >= 8 && hours <= 11) {
    periodTag = 'Morning Rush Peak';
    periodColor = 'text-amber-400 bg-amber-500/10 border-amber-500/30';
  } else if (hours >= 12 && hours <= 16) {
    periodTag = 'Midday Flow';
    periodColor = 'text-sky-400 bg-sky-500/10 border-sky-500/30';
  } else if (hours >= 17 && hours <= 21) {
    periodTag = 'Evening Gridlock Peak';
    periodColor = 'text-rose-400 bg-rose-500/10 border-rose-500/30';
  }

  // Playback timer
  useEffect(() => {
    if (!isPlaying) return;

    const intervalMs = 600 / speed;
    const timer = setInterval(() => {
      onSlotChange((slotIndex + 1) % 96);
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, slotIndex, speed, onSlotChange]);

  return (
    <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20 w-[92%] max-w-3xl px-5 py-3.5 rounded-2xl bg-slate-900/95 border border-slate-700/80 shadow-2xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-2">
        {/* Current Time Badge */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-slate-950 border border-slate-800 text-sky-400 font-mono font-bold text-sm">
            <Clock className="w-4 h-4 text-sky-400" />
            <span>Day 49 · {timeString}</span>
          </div>
          <span className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${periodColor}`}>
            {periodTag}
          </span>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center space-x-2">
          {/* Step Back */}
          <button
            onClick={() => onSlotChange(slotIndex > 0 ? slotIndex - 1 : 95)}
            title="Previous 15 mins"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          {/* Play/Pause */}
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            title={isPlaying ? 'Pause Playback' : 'Start Playback'}
            className="px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white flex items-center space-x-1.5 font-semibold text-xs shadow-md shadow-sky-600/30 transition-all cursor-pointer"
          >
            {isPlaying ? (
              <>
                <Pause className="w-3.5 h-3.5" />
                <span>Pause</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Play</span>
              </>
            )}
          </button>

          {/* Step Forward */}
          <button
            onClick={() => onSlotChange((slotIndex + 1) % 96)}
            title="Next 15 mins"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Speed Toggle */}
          <button
            onClick={() => setSpeed(speed === 1 ? 2 : speed === 2 ? 4 : 1)}
            title="Playback Speed"
            className="px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 flex items-center space-x-1"
          >
            <FastForward className="w-3 h-3 text-sky-400" />
            <span>{speed}x</span>
          </button>
        </div>
      </div>

      {/* Scrubbable Range Slider */}
      <div className="relative flex items-center w-full">
        <input
          type="range"
          min="0"
          max="95"
          step="1"
          value={slotIndex}
          onChange={(e) => onSlotChange(parseInt(e.target.value, 10))}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500 focus:outline-none"
        />
      </div>

      {/* Ticks representation */}
      <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
        <span>00:00</span>
        <span>06:00</span>
        <span className="text-amber-400 font-semibold">09:00 (Rush)</span>
        <span>12:00</span>
        <span className="text-rose-400 font-semibold">18:00 (Peak)</span>
        <span>23:45</span>
      </div>
    </div>
  );
};
