import "./index.css";
import { Composition } from "remotion";
import {
  ComparisonComposition,
  comparisonSchema,
  calculateComparisonMetadata,
} from "./Comparison";

const FPS = 60;
const HOLD_FRAMES = 30; // 0.5s — keep the final state briefly, don't drag

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Comparison"
      component={ComparisonComposition}
      schema={comparisonSchema}
      defaultProps={{
        fp16Path: "fp16.jsonl",
        tqPath: "tq.jsonl",
        metaPath: "meta.json",
        holdFrames: HOLD_FRAMES,
        speed: 3,
      }}
      calculateMetadata={calculateComparisonMetadata}
      durationInFrames={600}
      fps={FPS}
      width={1920}
      height={1080}
    />
  );
};
