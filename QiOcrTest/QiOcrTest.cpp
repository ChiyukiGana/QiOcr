#pragma once
#include <fstream>
#include <iostream>
#include <QiOcrInterface.h>

static bool readFile(const std::string& file, std::unique_ptr<char[]>& data, size_t& size)
{
	std::ifstream modelFile(file, std::ios::in | std::ios::binary | std::ios::ate);
	if (!modelFile) return false;

	size = modelFile.tellg();
	if (!size) return false;

	modelFile.seekg(0, std::ios::beg);
	data = std::make_unique<char[]>(size);
	modelFile.read(data.get(), size);
	return (bool)modelFile.gcount();
}

int main()
{
	std::locale::global(std::locale(".UTF8"));

	bool loadFromMemory = true;

	QiOcrInterface* ocr;
	if (loadFromMemory)
	{
		std::unique_ptr<char[]> rec;
		size_t recSize;
		if (!readFile("OCR\\ppocr.onnx", rec, recSize)) return -1;
		std::unique_ptr<char[]> keys;
		size_t keysSize;
		if (!readFile("OCR\\ppocr.keys", keys, keysSize)) return -1;
		std::unique_ptr<char[]> det;
		size_t detSize;
		if (!readFile("OCR\\ppdet.onnx", det, detSize)) return -1;

		ocr = QiOcrInterfaceInit(rec.get(), recSize, keys.get(), keysSize, det.get(), detSize);
	}
	else
	{
		ocr = QiOcrInterfaceInit();
	}
	if (!ocr)
	{
		std::cout << "OCR failed to init";
		return -1;
	}

	std::cout << "document mode:\n" << std::endl;
	{
		CImage image;
		image.Load(L"test.png");
		if (image.IsNull())
		{
			std::cout << "no image";
			return -1;
		}

		std::vector<std::string> result = ocr->scan_list(image);
		for (const std::string& i : result)
		{
			std::cout << i << std::endl;
		}
	}
	std::cout << "\n\nline mode:\n" << std::endl;
	{
		CImage image;
		image.Load(L"test2.png");
		if (image.IsNull())
		{
			std::cout << "no image";
			return -1;
		}

		std::cout << ocr->scan(image, true) << std::endl;
	}

	std::cout << "\n" << std::endl;
	system("pause");
	return 0;
}