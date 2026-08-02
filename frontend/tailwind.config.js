/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: "#1B365D",
          light: "#274B7D",
        },
        slate: {
          DEFAULT: "#4A607A",
        },
        ice: {
          50: "#F5F9FD",
          100: "#EBF2FA",
          200: "#D0E1F9",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
