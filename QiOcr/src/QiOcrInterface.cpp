#include <QiOcrInterface.h>
#include "QiOcr.h"

struct QiOcrInterfaceDef : QiOcrInterface
{
	std::vector<std::string> scan_list(const CImage& image)
	{
		return ocr->scan_list(image);
	}
	std::vector<std::string> scan_list(const RECT& rect_screen)
	{
		return ocr->scan_list(rect_screen);
	}
	std::string scan(const CImage& image)
	{
		return ocr->scan(image);
	}
	std::string scan(const RECT& rect_screen)
	{
		return ocr->scan(rect_screen);
	}
	QiOcrInterfaceDef() : ocr(new QiOcrTool())
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
	return nullptr;
}