"use client";

import React, { useEffect, useState } from 'react';
import mqtt, { MqttClient } from 'mqtt';

const BoardComponent: React.FC = () => {
  const [client, setClient] = useState<MqttClient | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    // Connect to the MQTT broker
    const mqttClient = mqtt.connect('ws://localhost:1884');

    mqttClient.on('connect', () => {
      console.log('Connected to MQTT broker');
      setIsConnected(true);
      // Subscribe to a topic
      mqttClient.subscribe('chess');
    });

    mqttClient.on('message', (topic: string, payload: Buffer) => {
      // Handle the received message
      console.log('Received Message:', topic, payload.toString());
      setMessage(payload.toString());
    });

    mqttClient.on('error', (err) => {
      console.error('Connection error: ', err);
      mqttClient.end();
    });

    mqttClient.on('close', () => {
      console.log('Connection closed');
      setIsConnected(false);
    });

    // Store the client in the state
    setClient(mqttClient);

    // Cleanup on unmount
    return () => {
      if (mqttClient) {
        mqttClient.end();
      }
    };
  }, []);

  return (
    <div style={{width:'600px'}}>
      <p>Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
      <div dangerouslySetInnerHTML={{ __html: message }} />
    </div>
  );
};

export default BoardComponent;