/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        graphite: {
          950: "#070A0F",
          900: "#0B1018",
          850: "#0F1621",
          800: "#141D2A",
          700: "#1F2B3B",
          600: "#2A394C"
        },
        signal: {
          cyan: "#67E8F9",
          emerald: "#34D399",
          amber: "#FBBF24",
          red: "#FB7185"
        }
      },
      boxShadow: {
        panel: "0 16px 44px rgba(0,0,0,0.24)"
      }
    }
  },
  plugins: []
};
