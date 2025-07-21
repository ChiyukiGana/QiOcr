#pragma once
#include <fstream>
#include <iostream>

#define QIOCR_SHARED

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

	bool speedTest = true;
	bool loadFromMemory = true;
	size_t threads = 0;

	size_t ver = QiOcrInterfaceVersion();

	std::cout << "qiocr version: " << ver << "\n" << std::endl;

	QiOcrModule ocr;
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

		ocr = QiOcrInterfaceInit(rec.get(), recSize, keys.get(), keysSize, det.get(), detSize, threads);
	}
	else
	{
		ocr = QiOcrInterfaceInit(threads);
	}
	if (!ocr.valid())
	{
		std::cout << "OCR failed to init";
		return -1;
	}

	if (speedTest)
	{
		CImage image;
		image.Load(L"test.png");
		if (image.IsNull())
		{
			std::cout << "no image";
			return -1;
		}
		while (true)
		{
			clock_t begin = clock();
			std::vector<std::string> result = ocr.scan_list(image);
			clock_t end = clock() - begin;

			std::cout << "time(ms): " << end << "\n" << std::endl;
		}
		return 0;
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

		clock_t begin = clock();
		std::vector<std::string> result = ocr.scan_list(image);
		clock_t end = clock() - begin;

		std::cout << "time(ms): " << end << "\n" << std::endl;

		std::cout << "------------------------------" << std::endl;
		for (const std::string& i : result)
		{
			std::cout << i << std::endl;
		}
		std::cout << "------------------------------" << std::endl;
	}
	std::cout << "\nline mode:\n" << std::endl;
	{
		CImage image;
		image.Load(L"test2.png");
		if (image.IsNull())
		{
			std::cout << "no image";
			return -1;
		}

		clock_t begin = clock();
		std::string result = ocr.scan(image, true);
		clock_t end = clock() - begin;

		std::cout << "time(ms): " << end << "\n" << std::endl;

		std::cout << "------------------------------" << std::endl;
		std::cout << result << std::endl;
		std::cout << "------------------------------" << std::endl;
	}

	std::cout << "\n" << std::endl;
	system("pause");
	return 0;
}