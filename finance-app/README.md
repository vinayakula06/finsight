# Finance App

A modern financial application built with Next.js, featuring portfolio management, stock analysis, AI trading signals, and real-time market updates.

## Features

- Portfolio Management
- Stock Analysis with AI Insights
- Real-time Market Data
- Interactive Charts
- AI Trading Signals
- Market News Integration
- Built-in Chat Assistant
- Real-time Notifications

## Tech Stack

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Tremor (for charts and UI components)
- Headless UI (for accessible components)

## Getting Started

1. Clone the repository:
```bash
git clone <repository-url>
cd finance-app
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
finance-app/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Home page
│   ├── portfolio/         # Portfolio section
│   ├── stocks/           # Stocks analysis section
│   └── layout.tsx        # Root layout
├── components/            # React components
│   ├── Navigation.tsx    # Main navigation
│   ├── Chatbot.tsx      # AI chat assistant
│   └── NotificationCenter.tsx # Notifications
├── public/               # Static assets
└── package.json         # Project dependencies
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
