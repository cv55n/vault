#include <QPixmap>
#include "OSD.h"

OSD::OSD(QWidget *parent) : QDialog(parent) {
    // esconde o timer osd
    hideOSDTimer = new QTimer(this);
    hideOSDTimer->setSingleShot(true);

    connect(hideOSDTimer, SIGNAL(timeout()), this, SLOT(hide()));

    // configurações da janela
    setWindowFlags(Qt::SplashScreen);
    setPalette(QPalette(QColor("darkBlue")));

    hboxLayout = new QHBoxLayout(this);

    icon = new QLabel(this);
    icon->setPixmap(QPixmap(":/Icons/qbittorrent16.png"));
    icon->adjustSize();

    msg = new QLabel(this);
    msg->setPalette(QPalette(QColor(88, 75, 255, 200)));
    icon->setPalette(QPalette(QColor(88, 75, 255, 200)));

    msg->setAutoFillBackground(true);
    icon->setAutoFillBackground(true);

    hboxLayout->addWidget(icon);
    hboxLayout->addWidget(msg);
    hboxLayout->setSpacing(0);
    hboxLayout->setMargin(1);
}

OSD::~OSD() {
    delete hideOSDTimer;
    delete icon;
    delete msg;
    delete hboxLayout;
}

void OSD::display(const QString& message) {
    if (hideOSDTimer->isActive()) {
        hideOSDTimer->stop();

        hide();
    }

    msg->setText("<font color='white'><b>"+message+"</b></font>");
    msg->adjustSize();

    adjustSize();
    show();
    
    hideOSDTimer->start(3000);
}