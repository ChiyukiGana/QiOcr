#include <QiOcrInterface.h>
#include "QiOcr.h"

struct QiOcrInterfaceDef : QiOcrInterface
{
	std::vector<std::string> scan_list(const CImage& image, bool skipDet = false)
	{
		return ocr->scan_list(image, skipDet);
	}
	std::vector<std::string> scan_list(const RECT& rect_screen, bool skipDet = false)
	{
		return ocr->scan_list(rect_screen, skipDet);
	}
	std::string scan(const CImage& image, bool skipDet = false)
	{
		return ocr->scan(image, skipDet);
	}
	std::string scan(const RECT& rect_screen, bool skipDet = false)
	{
		return ocr->scan(rect_screen, skipDet);
	}
	QiOcrInterfaceDef() : ocr(new QiOcrTool())
	{
	}
	QiOcrInterfaceDef(void* recData, size_t recSize, void* keyData, size_t keySize, void* detData, size_t detSize) : ocr(new QiOcrTool(recData, recSize, keyData, keySize, detData, detSize))
	{
	}
	~QiOcrInterfaceDef()
	{
		delete ocr;
	}
	QiOcrTool* ocr = nullptr;
};

extern "C" __declspec(dllexport) QiOcrInterface* _stdcall QiOcrInterfaceInitInterface()
{
	QiOcrInterfaceDef* ocr = new QiOcrInterfaceDef();
	if (ocr->ocr->isInit()) return (QiOcrInterface*)ocr;
	delete ocr;
	return nullptr;
}

extern "C" __declspec(dllexport) QiOcrInterface* _stdcall QiOcrInterfaceInitInterfaceFromMemory(void* recData, size_t recSize, void* keysData, size_t keysSize, void* detData, size_t detSize)
{
	QiOcrInterfaceDef* ocr = new QiOcrInterfaceDef(recData, recSize, keysData, keysSize, detData, detSize);
	if (ocr->ocr->isInit()) return (QiOcrInterface*)ocr;
	delete ocr;
	return nullptr;
}