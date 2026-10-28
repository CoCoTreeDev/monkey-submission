"use client";

import { useState } from "react";

type Message = { type: "success" | "error"; text: string };

export function useFlashMessage(duration = 4000) {
  const [message, setMessage] = useState<Message | null>(null);
  const flash = (type: Message["type"], text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), duration);
  };
  return [message, flash] as const;
}

export function MessageBanner({
  message,
  className = "",
}: {
  message: Message | null;
  className?: string;
}) {
  if (!message) return null;
  return (
    <div
      className={`${className ? `${className} ` : ""}rounded-md px-3 py-2 text-xs ${
        message.type === "success" ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"
      }`}
    >
      {message.text}
    </div>
  );
}
