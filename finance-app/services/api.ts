import axios from 'axios';

// Using environment variables for API keys
const FINNHUB_API_KEY = process.env.NEXT_PUBLIC_FINNHUB_API_KEY;
const ALPHA_VANTAGE_API_KEY = process.env.NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY;

if (!FINNHUB_API_KEY || !ALPHA_VANTAGE_API_KEY) {
  throw new Error('Missing API keys in environment variables');
}

const finnhubClient = axios.create({
  baseURL: 'https://finnhub.io/api/v1',
  params: {
    token: FINNHUB_API_KEY
  }
});

const alphaVantageClient = axios.create({
  baseURL: 'https://www.alphavantage.co/query',
  params: {
    apikey: ALPHA_VANTAGE_API_KEY
  }
});

export interface StockQuote {
  c: number; // Current price
  h: number; // High price of the day
  l: number; // Low price of the day
  o: number; // Open price of the day
  pc: number; // Previous close price
  t: number; // Timestamp
}

export interface CompanyProfile {
  name: string;
  ticker: string;
  marketCap: number;
  industry: string;
  peRatio: number;
  website: string;
}

export const api = {
  // Stock quote
  getStockQuote: async (symbol: string): Promise<StockQuote> => {
    try {
      const response = await finnhubClient.get(`/quote`, {
        params: { symbol: symbol.toUpperCase() }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching stock quote:', error);
      throw error;
    }
  },

  // Company profile
  getCompanyProfile: async (symbol: string): Promise<CompanyProfile> => {
    try {
      const response = await finnhubClient.get(`/stock/profile2`, {
        params: { symbol: symbol.toUpperCase() }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching company profile:', error);
      throw error;
    }
  },

  // Historical data
  getHistoricalData: async (symbol: string, interval: string = 'D') => {
    try {
      const response = await alphaVantageClient.get('', {
        params: {
          function: 'TIME_SERIES_DAILY',
          symbol: symbol.toUpperCase(),
          outputsize: 'compact'
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching historical data:', error);
      throw error;
    }
  },

  // Market news
  getMarketNews: async () => {
    try {
      const response = await finnhubClient.get('/news', {
        params: {
          category: 'general'
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching market news:', error);
      throw error;
    }
  },

  // Portfolio data (mock)
  getPortfolioData: async () => {
    // This would typically come from your backend
    return {
      totalValue: 156789.42,
      dailyChange: 1234.56,
      dailyChangePercent: 2.34,
      holdings: [
        { symbol: 'AAPL', shares: 100, avgPrice: 150.00, currentPrice: 175.23 },
        { symbol: 'GOOGL', shares: 50, avgPrice: 2500.00, currentPrice: 2750.45 },
        { symbol: 'MSFT', shares: 75, avgPrice: 200.00, currentPrice: 225.67 },
        { symbol: 'AMZN', shares: 25, avgPrice: 3000.00, currentPrice: 3250.89 }
      ],
      history: [
        { date: '2024-01', value: 145000 },
        { date: '2024-02', value: 148000 },
        { date: '2024-03', value: 152000 },
        { date: '2024-04', value: 156789.42 }
      ]
    };
  }
}; 