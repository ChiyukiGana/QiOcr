#pragma once
#include <fstream>
#include <iostream>

#define QIOCR_SHARED

#include <QiOcrInterface.h>

static std::vector<char> readFile(const std::wstring& file)
{
	HANDLE hFile = CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	if (hFile && hFile != INVALID_HANDLE_VALUE)
	{
		DWORD size = GetFileSize(hFile, NULL);
		if (size && size != INVALID_FILE_SIZE)
		{
			DWORD bytesReaded = 0;
			std::vector<char> data(size);
			BOOL b = ReadFile(hFile, data.data(), size, &bytesReaded, 0);
			CloseHandle(hFile);
			if (b != FALSE) return data;
		}
		CloseHandle(hFile);
	}
	return std::vector<char>();
}

int main()
{
	std::locale::global(std::locale(".UTF8"));

	bool speedTest = false;
	bool loadFromMemory = false;
	size_t threads = 0;

	size_t ver = QiOcrVersion(L"qiocr.dll");

	std::cout << "qiocr version: " << ver << "\n" << std::endl;
	if (threads) std::cout << "threads: " << threads << "\n" << std::endl;
	else std::cout << "threads: auto" << "\n" << std::endl;

	QiOcrModule ocr;
	if (loadFromMemory)
	{
		std::vector<char> rec = readFile(L"OCR\\ppocr.onnx");
		std::vector<char> keys = readFile(L"OCR\\ppocr.keys");
		std::vector<char> det = readFile(L"OCR\\ppdet.onnx");
		ocr = QiOcrInit(L"qiocr.dll", rec.data(), rec.size(), keys.data(), keys.size(), det.data(), det.size(), threads);
	}
	else
	{
		ocr = QiOcrInit(L"qiocr.dll", L"OCR\\ppocr.onnx", L"OCR\\ppocr.keys", L"OCR\\ppdet.onnx", threads);
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

		std::vector<POINT> centers;

		clock_t begin = clock();
		std::vector<std::string> result = ocr.scan_list(image, false, &centers);
		clock_t end = clock() - begin;

		std::cout << "time(ms): " << end << "\n" << std::endl;

		std::cout << "------------------------------" << std::endl;
		for (size_t i = 0; i < result.size(); i++)
		{
			std::string s = result[i];
			POINT p = centers[i];
			std::cout << p.x << "-" << p.y << ": ";
			std::cout << s << std::endl;
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