import { createContext } from "react";
import { ValueContextState } from "../common/event";

interface AnimationContextState extends ValueContextState {
};

export const AnimationContext = createContext<AnimationContextState>({
	setValue: (type: string, val: string | number) => { }
});