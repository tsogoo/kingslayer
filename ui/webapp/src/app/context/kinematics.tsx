import { createContext } from "react";

export interface Arm {
  L1: number;
  L2: number;
  position: Point;
  color: string;
  opposite?: boolean;
};

export interface Point {
  x: number;
  y: number;
};

export interface KinematicsContextState {
  position: Point;
  offset: Point;
};

export const x0y0:Point = {x:0,y:0};

export const KinematicsContext = createContext<KinematicsContextState>({
  position:x0y0,
  offset:x0y0
});