#pragma once
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <fstream>
#include <windows.h>
#include <atlimage.h>

#include <onnxruntime_cxx_api.h>
#pragma comment(lib,"onnxruntime.lib")

#include <opencv2/opencv.hpp>
#pragma comment(lib,"opencv_core4110.lib")
#pragma comment(lib,"opencv_imgproc4110.lib")
#pragma comment(lib,"zlib.lib")

#ifndef AlignmentSize
#define AlignmentSize(size, alignment) ((alignment > 1) ? ((size%alignment) ? (size+(alignment-(size%alignment))) : size) : size)
#endif

struct OnnxOcrResult
{
	enum
	{
		r_ok,
		r_model_notfound,
		r_keys_notfound,
		r_model_invalid,
		r_keys_invalid,
		r_sdk_different
	};
};

class OcrBase
{
protected:
	static constexpr float s_meanValue = 127.5f;
	static constexpr float s_normValue = 1.0 / s_meanValue;
	std::unique_ptr<Ort::Env> m_env;
	std::unique_ptr<Ort::Session> m_session;
	char* m_inputName = nullptr;
	char* m_outputName = nullptr;
	bool m_init = false;
public:
	static std::string toString(std::wstring val, UINT codePage = CP_UTF8)
	{
		int length = WideCharToMultiByte(codePage, 0, val.c_str(), val.size(), 0, 0, 0, 0);
		if (length)
		{
			std::string result(length, 0);
			length = WideCharToMultiByte(codePage, 0, val.c_str(), val.size(), &result[0], length, 0, 0);
			if (length) return result;
		}
		return std::string();
	}
	static std::wstring toWString(std::string val, DWORD codePage = CP_UTF8)
	{
		int length = MultiByteToWideChar(codePage, 0, val.c_str(), val.size(), 0, 0);
		if (length)
		{
			std::wstring result(length, 0);
			length = MultiByteToWideChar(codePage, 0, val.c_str(), val.size(), &result[0], length);
			if (length) return result;
		}
		return std::wstring();
	}
public:
	~OcrBase()
	{
		release();
	}
	virtual bool isInit() const
	{
		return m_init;
	}
	virtual void release()
	{
		m_init = false;
		if (m_inputName)
		{
			free(m_inputName);
			m_inputName = nullptr;
		}
		if (m_outputName)
		{
			free(m_outputName);
			m_outputName = nullptr;
		}
	}
	virtual std::vector<float> makeTensorValues(const cv::Mat& src)
	{
		if (src.empty()) return std::vector<float>();

		int width = src.cols;
		int height = src.rows;
		int numChannels = src.channels();

		if (numChannels < 3)
		{
			cv::Mat bgrImage;
			cv::cvtColor(src, bgrImage, cv::COLOR_GRAY2BGR);
			return makeTensorValues(bgrImage);
		}

		size_t imageSize = width * height;
		std::vector<float> inputTensorValues(imageSize * 3);

		cv::Mat bgrImage;
		if (numChannels == 4) cv::cvtColor(src, bgrImage, cv::COLOR_BGRA2BGR);
		else bgrImage = src;

		for (int y = 0; y < height; ++y) {
			const cv::Vec3b* row = bgrImage.ptr<cv::Vec3b>(y);
			for (int x = 0; x < width; ++x) {
				const cv::Vec3b& pixel = row[x];

				BYTE blue = pixel[0];
				BYTE green = pixel[1];
				BYTE red = pixel[2];

				float normRed = (static_cast<float>(red) - s_meanValue) * s_normValue;
				float normGreen = (static_cast<float>(green) - s_meanValue) * s_normValue;
				float normBlue = (static_cast<float>(blue) - s_meanValue) * s_normValue;

				size_t index = y * width + x;
				inputTensorValues[index] = normRed;
				inputTensorValues[imageSize + index] = normGreen;
				inputTensorValues[2 * imageSize + index] = normBlue;
			}
		}

		return inputTensorValues;
	}
};

class OcrDet : public OcrBase
{
public:
	int init(const std::string& model, size_t threads = 2)
	{
		OcrBase::release();
		if (!threads) threads = 1;

		std::ifstream modelFile(model);
		if (!modelFile) return OnnxOcrResult::r_model_notfound;

		if (!Ort::Global<void>::api_) return OnnxOcrResult::r_sdk_different;
		try
		{
			m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "OnnxOcrDet");
		}
		catch (...)
		{
			return OnnxOcrResult::r_sdk_different;
		}
		try
		{
			Ort::SessionOptions options;
			options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			options.SetInterOpNumThreads(threads);

			m_session = std::make_unique<Ort::Session>(*m_env, toWString(model).c_str(), options);

			size_t inputCount = m_session->GetInputCount();
			if (!inputCount) return OnnxOcrResult::r_model_invalid;

			Ort::AllocatorWithDefaultOptions allocator;
			Ort::AllocatedStringPtr s = m_session->GetInputNameAllocated(0, allocator);
			m_inputName = strdup(s.get());

			s = m_session->GetOutputNameAllocated(0, allocator);
			m_outputName = strdup(s.get());
		}
		catch (...)
		{
			return OnnxOcrResult::r_model_invalid;
		}

		m_init = true;
		return OnnxOcrResult::r_ok;
	}

	std::vector<cv::Mat> scan(const cv::Mat& image, float margin_ratio = 1.0f)
	{
		if (!isInit()) return std::vector<cv::Mat>();
		if (image.empty()) return std::vector<cv::Mat>();
		if (image.channels() < 3) return std::vector<cv::Mat>();

		cv::Mat imageScaled = resizeImage(image, 32);
		std::vector<float> tensorValues = OcrBase::makeTensorValues(imageScaled);
		std::vector<int64_t> inputShape{ 1, 3, imageScaled.rows, imageScaled.cols };
		try
		{
			Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

			Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, tensorValues.data(), tensorValues.size(), inputShape.data(), inputShape.size());
			if (!inputTensor.IsTensor()) return std::vector<cv::Mat>();

			std::vector<Ort::Value> outputTensor = m_session->Run(Ort::RunOptions{}, &m_inputName, &inputTensor, 1, &m_outputName, 1);
			if (outputTensor.empty() || outputTensor.size() != 1 || !outputTensor.front().IsTensor()) return std::vector<cv::Mat>();

			std::vector<int64_t> outputShape = outputTensor.front().GetTensorTypeAndShapeInfo().GetShape();
			if (outputShape.size() != 4 || outputShape[0] != 1 || outputShape[1] != 1) return std::vector<cv::Mat>();

			int64_t outputHeight = outputShape[2];
			int64_t outputWidth = outputShape[3];
			float* floatArray = outputTensor.front().GetTensorMutableData<float>();

			cv::Mat outputMat(outputHeight, outputWidth, CV_32F, floatArray);

			cv::Mat binaryMat;
			double thresholdValue = 0.3;
			cv::threshold(outputMat, binaryMat, thresholdValue, 1.0, cv::THRESH_BINARY);
			binaryMat.convertTo(binaryMat, CV_8U, 255);

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(binaryMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			std::vector<cv::Rect> boxes;

			for (const auto& contour : contours) {
				cv::Rect rect = cv::boundingRect(contour);
				if (rect.area() < 24) continue;

				int margin = std::round(rect.height * margin_ratio);
				int new_x = std::max(0, rect.x - margin);
				int new_y = std::max(0, rect.y - margin);
				int new_width = rect.width + 2 * margin;
				int new_height = rect.height + 2 * margin;

				new_width = std::min(new_width, imageScaled.cols - new_x);
				new_height = std::min(new_height, imageScaled.rows - new_y);

				if (new_width <= 0 || new_height <= 0) continue;

				boxes.emplace_back(new_x, new_y, new_width, new_height);
			}

			std::vector<cv::Mat> regions;
			for (std::vector<cv::Rect>::const_reverse_iterator i = boxes.rbegin(); i != boxes.rend(); i++)
			{
				const cv::Rect& rect = *i;
				if (rect.width > 0 && rect.height > 0) regions.emplace_back(imageScaled(rect).clone());
			}

			return regions;
		}
		catch (...)
		{
			return std::vector<cv::Mat>();
		}
	}

	static cv::Mat resizeImage(const cv::Mat& srcImage, size_t alignment = 32)
	{
		if (srcImage.empty()) return srcImage;

		int dstWidth = AlignmentSize(srcImage.cols, alignment);
		int dstHeight = AlignmentSize(srcImage.rows, alignment);

		if (dstWidth == srcImage.cols && dstHeight == srcImage.rows) return srcImage.clone();

		cv::Mat dstImage;
		cv::resize(srcImage, dstImage, cv::Size(dstWidth, dstHeight), 0, 0, cv::INTER_LINEAR);

		return dstImage;
	}
};

class OcrRec : public OcrBase
{
	std::vector<std::string> m_keys;
	size_t m_scaleSize = 48;
public:
	int init(const std::string& model, const std::string& keys, size_t threads = 2, size_t scaleSize = 48)
	{
		OcrBase::release();
		m_keys.clear();
		if (!threads) threads = 1;
		m_scaleSize = scaleSize;

		std::ifstream modelFile(model);
		if (!modelFile) return OnnxOcrResult::r_model_notfound;

		std::ifstream keysFile(keys);
		if (!keysFile) return OnnxOcrResult::r_keys_notfound;

		std::string line;
		while (std::getline(keysFile, line)) m_keys.push_back(line);

		if (m_keys.empty()) return OnnxOcrResult::r_keys_invalid;

		if (!Ort::Global<void>::api_) return OnnxOcrResult::r_sdk_different;
		try
		{
			m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "OnnxOcrRec");
		}
		catch (...)
		{
			return OnnxOcrResult::r_sdk_different;
		}
		try
		{
			Ort::SessionOptions options;
			options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			options.SetInterOpNumThreads(threads);

			m_session = std::make_unique<Ort::Session>(*m_env, toWString(model).c_str(), options);

			size_t inputCount = m_session->GetInputCount();
			if (!inputCount) return OnnxOcrResult::r_model_invalid;

			Ort::AllocatorWithDefaultOptions allocator;
			Ort::AllocatedStringPtr s = m_session->GetInputNameAllocated(0, allocator);
			m_inputName = strdup(s.get());

			s = m_session->GetOutputNameAllocated(0, allocator);
			m_outputName = strdup(s.get());
		}
		catch (...)
		{
			return OnnxOcrResult::r_model_invalid;
		}

		m_init = true;
		return OnnxOcrResult::r_ok;
	}

	std::string scoreToString(const std::vector<float>& outputData, int h, int w)
	{
		std::string result;
		int indexPrev = 0;
		for (int i = 0; i < h; i++) {
			int firstChar = i * w;
			int lastChar = i * w + w;
			int index = std::distance(outputData.begin() + firstChar, std::max_element(outputData.begin() + firstChar, outputData.begin() + lastChar));
			if (index > 0 && index <= m_keys.size() && index != indexPrev) result += m_keys[index - 1];
			indexPrev = index;
		}
		return result;
	}

	std::string scan(const cv::Mat& image) {
		if (!isInit()) return std::string();
		if (image.empty()) return std::string();
		if (image.channels() < 3) return std::string();

		cv::Mat imageScaled = resizeWithHeight(image, m_scaleSize);
		std::vector<float> tensorValues = OcrBase::makeTensorValues(imageScaled);
		std::vector<int64_t> inputShape{ 1, 3, imageScaled.rows, imageScaled.cols };

		try
		{
			Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

			Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, tensorValues.data(), tensorValues.size(), inputShape.data(), inputShape.size());
			if (!inputTensor.IsTensor()) return std::string();

			std::vector<Ort::Value> outputTensor = m_session->Run(Ort::RunOptions{}, &m_inputName, &inputTensor, 1, &m_outputName, 1);
			if (outputTensor.empty() || outputTensor.size() != 1 || !outputTensor.front().IsTensor()) return std::string();

			std::vector<int64_t> outputShape = outputTensor.front().GetTensorTypeAndShapeInfo().GetShape();
			int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());

			float* floatArray = outputTensor.front().GetTensorMutableData<float>();
			return scoreToString(std::vector<float>(floatArray, floatArray + outputCount), outputShape[1], outputShape[2]);
		}
		catch (...)
		{
			return std::string();
		}
	}

	static cv::Mat resizeWithHeight(const cv::Mat& srcImage, size_t height) {
		double scaleFactor = static_cast<double>(height) / srcImage.rows;
		int newWidth = static_cast<int>(srcImage.cols * scaleFactor);
		cv::Mat destImage;
		cv::resize(srcImage, destImage, cv::Size(newWidth, height), 0, 0, cv::INTER_LINEAR);
		return destImage;
	}
};

class QiOcrTool
{
	class OcrDet* det;
	class OcrRec* rec;
public:
	QiOcrTool() : rec(new OcrRec), det(new OcrDet)
	{
		SYSTEM_INFO info; GetSystemInfo(&info);
		size_t threads = info.dwNumberOfProcessors >> 1;
		if (threads < 2) threads = 2;

		for (size_t i = 0; i < 2; i++)
		{
			int result;
			std::wstring title;
			if (i)
			{
				result = rec->init("OCR\\ppocr.onnx", "OCR\\ppocr.keys", threads, 48);
				title = L"OCR识别初始化错误";
			}
			else
			{
				result = det->init("OCR\\ppdet.onnx", threads);
				title = L"OCR检测初始化错误";
			}
			switch (result)
			{
			case OnnxOcrResult::r_ok: continue;
			case OnnxOcrResult::r_model_notfound: MessageBoxW(nullptr, L"模型不存在", title.c_str(), MB_ICONERROR); return;
			case OnnxOcrResult::r_keys_notfound: MessageBoxW(nullptr, L"Key不存在", title.c_str(), MB_ICONERROR); return;
			case OnnxOcrResult::r_model_invalid: MessageBoxW(nullptr, L"模型无效", title.c_str(), MB_ICONERROR); return;
			case OnnxOcrResult::r_keys_invalid: MessageBoxW(nullptr, L"Key无效", title.c_str(), MB_ICONERROR); return;
			case OnnxOcrResult::r_sdk_different: MessageBoxW(nullptr, L"Sdk不一致", title.c_str(), MB_ICONERROR); return;
			default: MessageBoxW(nullptr, L"未知错误", title.c_str(), MB_ICONERROR); return;
			}
		}
	}

	~QiOcrTool()
	{
		delete rec;
		delete det;
	}

	bool isInit() const
	{
		return det->isInit() && rec->isInit();
	}

	std::vector<std::string> scan_list(const CImage& image)
	{
		if (!isInit()) return std::vector<std::string>();
		cv::Mat mat = toMat(image);
		if (mat.empty()) return std::vector<std::string>();

		std::vector<cv::Mat> textBlock = det->scan(mat, 1.0f);
		std::vector<std::string> result;
		for (const cv::Mat& i : textBlock)
		{
			std::string text = rec->scan(i);
			if (text.empty()) continue;
			result.push_back(text);
		}
		return result;
	}

	std::vector<std::string> scan_list(const RECT& rect)
	{
		if (!isInit()) return std::vector<std::string>();
		int w = rect.right - rect.left;
		int h = rect.bottom - rect.top;
		std::vector<std::string> result;
		if (w > 0 && h > 0)
		{
			CImage image; image.Create(w, h, 32);
			HDC hdc = GetDC(nullptr);
			if (BitBlt(image.GetDC(), 0, 0, w, h, hdc, rect.left, rect.top, SRCCOPY)) result = scan_list(image);
			image.ReleaseDC();
			ReleaseDC(nullptr, hdc);
		}
		return result;
	}

	std::string scan(const CImage& image)
	{
		std::vector<std::string> result = scan_list(image);
		std::string text;
		for (const std::string& i : result)
		{
			if (text.empty()) text += i;
			else text += std::string("\t") + i;
		}
		return text;
	}

	std::string scan(const RECT& rect)
	{
		std::vector<std::string> result = scan_list(rect);
		std::string text;
		for (const std::string& i : result)
		{
			if (text.empty()) text += i;
			else text += std::string("\t") + i;
		}
		return text;
	}

	static cv::Mat toMat(const CImage& image)
	{
		int width = image.GetWidth();
		int height = image.GetHeight();
		int bpp = image.GetBPP();
		int pitch = image.GetPitch();
		if (bpp != 24 && bpp != 32) return cv::Mat();

		cv::Mat mat(height, width, CV_8UC3);
		BYTE* pBits = (BYTE*)image.GetBits();
		for (int y = 0; y < height; ++y)
		{
			BYTE* pRow = pBits + y * pitch;

			if (bpp == 32)
			{
				for (int x = 0; x < width; ++x)
				{
					mat.at<cv::Vec3b>(y, x) = cv::Vec3b(pRow[x * 4 + 0], pRow[x * 4 + 1], pRow[x * 4 + 2]);
				}
			}
			else
			{
				for (int x = 0; x < width; ++x)
				{
					mat.at<cv::Vec3b>(y, x) = cv::Vec3b(pRow[x * 3 + 0], pRow[x * 3 + 1], pRow[x * 3 + 2]);
				}
			}
		}
		return mat;
	}
};