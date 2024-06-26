import * as ort from 'onnxruntime-web';
import ModelCache from './ModelCache';

class InferenceService {
    private modelPath: string;
    private metadataPath: string;
    private metadata: any;
    private modelCache: ModelCache;
    private session: ort.InferenceSession | null;

    constructor(modelPath: string, metadataPath: string) {
        this.modelPath = modelPath;
        this.metadataPath = metadataPath;
        this.session = null;
        this.modelCache = new ModelCache();
    }

    async loadModel(): Promise<void> {
        try {

            const modelBuffer = await this.modelCache.loadModel(this.modelPath, "trial");
            this.session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ['webgl'] });
            this.metadata = await fetch(this.metadataPath).then((response) => response.json());
        } catch (error) {
            throw new Error('Failed to load model: ' + (error as Error).message);
        }
    }

    async preprocessImage(image: HTMLImageElement): Promise<ort.Tensor> {
        const canvas = document.createElement('canvas');
        const size = this.metadata["size"] ?? 384;
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get 2D context');
        }
        ctx.drawImage(image, 0, 0, size, size);

        const imageData = ctx.getImageData(0, 0, size, size);
        const { data } = imageData;

        const mean = this.metadata["mean"] ?? [0.485, 0.456, 0.406];
        const std = this.metadata["std"] ?? [0.229, 0.224, 0.225];

        const float32Array = new Float32Array(size * size * 3);
        // color channel
        for (let c = 0; c < 3; c++) {
            // height
            for (let h = 0; h < size; h++) {
                // width
                for (let w = 0; w < size; w++) {
                    const idx = h * size + w;
                    float32Array[c * size * size + idx] = (data[idx * 4 + c] / 255.0 - mean[c]) / std[c];
                }
            }
        }

        // TODO: fromImage could be used if normalization happens in the model...
        // const output = await ort.Tensor.fromImage(imageData);

        return new ort.Tensor('float32', float32Array, [1, 3, 384, 384]);
    }

    async runInference(image: HTMLImageElement): Promise<Record<string, any>> {
        if (!this.session) {
            throw new Error('Model not loaded');
        }

        const tensor = await this.preprocessImage(image);

        //console.log("First 10 input samples: ", tensor.data.slice(0, 10));

        const feeds = { input: tensor };
        const output = await this.session.run(feeds);

        const outArray = output['output'].data as Float32Array;
        const values = Array.from(outArray);

        // softmax
        const exped = values.map((x: number) => Math.exp(x));
        const sum = exped.reduce((a: number, b: number) => a + b, 0);
        const softmax = exped.map((x: number) => x / sum);

        //console.log("First 10 output samples: ", softmax.slice(0, 10));

        let maxIdx = 0;

        for (let i = 1; i < softmax.length; i++) {
            if (softmax[i] > softmax[maxIdx]) {
                maxIdx = i;
            }
        }

        console.log("Max index: ", maxIdx);
        console.log("Max logit: ", values[maxIdx]);
        console.log("Exp sum: ", sum);


        return this._createResults(softmax, 5);
    }

    _createResults(probs: Array<number>, topK: number = 5, minProb: number = 0.00001): Record<string, any> {

        const labels = this.metadata["labels"];

        // choose top k indices from output:
        let indexed = probs.map((prob, index) => ({index, prob}));
        indexed.sort((a, b) => b.prob - a.prob);

        // filter out low-probability results:
        indexed = indexed.filter(i => i.prob >= minProb);

        let topKIndices = indexed.slice(0, topK).map(i => i.index);
        let topKClasses = topKIndices.map(i => labels[i]);

        let infos = this.metadata["infos"];
        let topKData = topKIndices.map(i => ({ info: infos[labels[i]], probability: probs[i] }));


        let zipped: Record<string, any> = {};
        topKClasses.forEach((className, index) => {
            zipped[className] = topKData[index];
        });


        return zipped;

    }
}

export default InferenceService;
