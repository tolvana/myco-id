import * as ort from 'onnxruntime-web';
import ModelCache from './ModelCache';

class ImageClassifier {
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

    async load(onProgress: (progress: number) => void): Promise<void> {
        try {
            const modelName = "trial";
            const modelBuffer = await this.modelCache.loadModel(this.modelPath, modelName, onProgress);
            this.metadata = await this.modelCache.loadMetadata(this.metadataPath, modelName);

            ort.env.wasm.proxy = true;
            this.session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ['wasm'] });
        } catch (error) {
            throw new Error('Failed to load model: ' + (error as Error).message);
        }
    }

    async preprocessImage(image: HTMLImageElement): Promise<ort.Tensor> {
        const size = this.metadata["size"] ?? 384;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get 2D context');
        }

        // Calculate dimensions for cropping
        const aspect = image.width / image.height;
        let targetWidth: number;
        let targetHeight: number;
        let offsetX: number;
        let offsetY: number;

        console.log("image.width: ", image.width);
        console.log("image.height: ", image.height);

        if (aspect > 1) {
            // Landscape orientation: width is the larger dimension
            targetWidth = image.width * (size / image.height);
            targetHeight = size;
            offsetX = (image.width - image.height) / 2;
            offsetY = 0;
        } else {
            // Portrait orientation: height is the larger dimension
            targetWidth = size;
            targetHeight = image.height * (size / image.width);
            offsetX = 0;
            offsetY = (image.height - image.width) / 2;
        }
        console.log("targetWidth: ", targetWidth);
        console.log("targetHeight: ", targetHeight);
        console.log("offsetX: ", offsetX);
        console.log("offsetY: ", offsetY);
        console.log("size: ", size);

        // Draw the image onto the canvas with cropping
        ctx.drawImage(image, offsetX, offsetY, image.width, image.height, 0, 0, targetWidth, targetHeight);

        const imageData = ctx.getImageData(0, 0, size, size);
        const { data } = imageData;

        return await ort.Tensor.fromImage(imageData);
    }

    async classify(image: HTMLImageElement): Promise<Record<string, any>> {
        if (!this.session) {
            throw new Error('Model not loaded');
        }

        let start = performance.now()
        const tensor = await this.preprocessImage(image);
        console.log(performance.now() - start)

        //console.log("First 10 input samples: ", tensor.data.slice(0, 10));

        const feeds = { input: tensor };

        start = performance.now()
        const output = await this.session.run(feeds);
        console.log(performance.now() - start)

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

    _createResults(probs: Array<number>, topK: number = 5, minProb: number = 0.001): Record<string, any> {

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

export default ImageClassifier;
