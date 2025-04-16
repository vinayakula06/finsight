'use client'

import { Fragment, useState } from 'react'
import { Menu, Transition } from '@headlessui/react'
import { BellIcon } from '@heroicons/react/24/outline'

// Sample notifications - replace with real data in production
const initialNotifications = [
  {
    id: 1,
    title: 'Portfolio Alert',
    message: 'Your portfolio value increased by 5%',
    time: '1 hour ago',
    type: 'success',
  },
  {
    id: 2,
    title: 'Market Update',
    message: 'Major market indices showing volatility',
    time: '2 hours ago',
    type: 'warning',
  },
  {
    id: 3,
    title: 'Trade Executed',
    message: 'Buy order for AAPL executed successfully',
    time: '3 hours ago',
    type: 'info',
  },
]

export default function NotificationCenter() {
  const [notifications, setNotifications] = useState(initialNotifications)
  const [unreadCount, setUnreadCount] = useState(2)

  const clearNotifications = () => {
    setNotifications([])
    setUnreadCount(0)
  }

  const markAllAsRead = () => {
    setUnreadCount(0)
  }

  return (
    <Menu as="div" className="relative">
      <Menu.Button className="relative rounded-full p-2 text-gray-400 hover:text-gray-500">
        <span className="sr-only">View notifications</span>
        <BellIcon className="h-6 w-6" aria-hidden="true" />
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 -mt-1 -mr-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-xs font-medium text-white">
            {unreadCount}
          </span>
        )}
      </Menu.Button>

      <Transition
        as={Fragment}
        enter="transition ease-out duration-100"
        enterFrom="transform opacity-0 scale-95"
        enterTo="transform opacity-100 scale-100"
        leave="transition ease-in duration-75"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-95"
      >
        <Menu.Items className="absolute right-0 z-10 mt-2 w-80 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
          <div className="px-4 py-2 border-b border-gray-100">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium text-gray-900">Notifications</h2>
              <div className="flex space-x-2">
                <button
                  onClick={markAllAsRead}
                  className="text-xs text-blue-600 hover:text-blue-800"
                >
                  Mark all as read
                </button>
                <button
                  onClick={clearNotifications}
                  className="text-xs text-gray-500 hover:text-gray-700"
                >
                  Clear all
                </button>
              </div>
            </div>
          </div>

          <div className="max-h-96 overflow-y-auto">
            {notifications.length === 0 ? (
              <div className="px-4 py-6 text-center text-sm text-gray-500">
                No notifications
              </div>
            ) : (
              notifications.map((notification) => (
                <Menu.Item key={notification.id}>
                  {({ active }) => (
                    <div
                      className={`
                        px-4 py-3 border-b border-gray-100 last:border-0
                        ${active ? 'bg-gray-50' : ''}
                      `}
                    >
                      <div className="flex items-start">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900">
                            {notification.title}
                          </p>
                          <p className="text-sm text-gray-500">
                            {notification.message}
                          </p>
                          <p className="mt-1 text-xs text-gray-400">
                            {notification.time}
                          </p>
                        </div>
                        <div className="ml-3 flex-shrink-0">
                          <div
                            className={`h-2 w-2 rounded-full ${
                              notification.type === 'success'
                                ? 'bg-green-400'
                                : notification.type === 'warning'
                                ? 'bg-yellow-400'
                                : 'bg-blue-400'
                            }`}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </Menu.Item>
              ))
            )}
          </div>
        </Menu.Items>
      </Transition>
    </Menu>
  )
} 