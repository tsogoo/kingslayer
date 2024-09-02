"use client";

import React, { useEffect, useState } from 'react';
import mqtt, { MqttClient } from 'mqtt';

interface BoardData {
  svg: string;
  move: string;
  history: string[];
}

const BoardComponent: React.FC = () => {
  const [client, setClient] = useState<MqttClient | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [svg, setSvg] = useState('');
  const [move, setMove] = useState('');
  const [history, setHistory] = useState<string[]>([]);

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
      const data: BoardData = JSON.parse(payload.toString());
      console.log('Received Message:', topic, data);
      setSvg(data.svg);
      setMove(data.move);
      setHistory(data.history.reverse());
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
    <div>
      <p>Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
      <div style={{float:'left', width:'600px'}} dangerouslySetInnerHTML={{ __html: svg }} />
      <div style={{float:'left', padding:'0 0 0 1em'}}>
        <div>last move: { move }</div>
        <div>{history.map((h, i) => (
          <div key={i}>{i+1}. {move == h ? <span style={{color:'white',background:'black'}}>{h}</span> : h}</div>
        ))}</div>
      </div>
    </div>
  );
};

export default BoardComponent;