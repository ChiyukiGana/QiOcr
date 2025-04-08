#pragma once
#include <iostream>
#include <QiOcrInterface.h>

int main()
{
	std::locale::global(std::locale(".UTF8"));

	QiOcrInterface* ocr = QiOcrInterfaceInit();
	if (!ocr)
	{
		std::cout << "OCR failed to init";
		return -1;
	}

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

	system("pause");
	return 0;
}