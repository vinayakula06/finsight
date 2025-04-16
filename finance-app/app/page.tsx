'use client'

import Navigation from '@/components/Navigation'
import Chatbot from '@/components/Chatbot'
import { Card, Title, Text } from '@tremor/react'

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="lg:pl-72">
        <div className="px-4 py-10 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            <Card>
              <Title>Portfolio Overview</Title>
              <Text>View and manage your investment portfolio</Text>
            </Card>
            
            <Card>
              <Title>Market Insights</Title>
              <Text>Get real-time market analysis and trends</Text>
            </Card>
            
            <Card>
              <Title>AI Trading Signals</Title>
              <Text>Receive AI-powered trading recommendations</Text>
            </Card>
            
            <Card>
              <Title>Latest News</Title>
              <Text>Stay updated with market news and events</Text>
            </Card>
            
            <Card>
              <Title>Performance Analytics</Title>
              <Text>Track your investment performance</Text>
            </Card>
            
            <Card>
              <Title>Smart Assistant</Title>
              <Text>Get help with your investment decisions</Text>
            </Card>
          </div>
        </div>
      </main>

      <Chatbot />
    </div>
  )
}
