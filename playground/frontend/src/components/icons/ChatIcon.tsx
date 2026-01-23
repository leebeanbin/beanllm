export function ChatIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Modern chat bubble with gradient effect */}
      <defs>
        <linearGradient id="chatGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.3" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.1" />
        </linearGradient>
      </defs>
      
      {/* Main chat bubble */}
      <path
        d="M20 2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h4l4 4 4-4h4c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"
        fill="url(#chatGradient)"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeOpacity="0.2"
      />
      
      {/* Sparkle/star decoration */}
      <path
        d="M12 8l1.5 3 3 0.5-2.25 2.25 0.5 3L12 15.5 9.25 17l0.5-3L7.5 11.5l3-0.5L12 8z"
        fill="currentColor"
        opacity="0.4"
      />
      
      {/* Subtle dots */}
      <circle cx="9" cy="10" r="1" fill="currentColor" opacity="0.3" />
      <circle cx="15" cy="10" r="1" fill="currentColor" opacity="0.3" />
    </svg>
  );
}
